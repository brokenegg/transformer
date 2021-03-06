# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Download and preprocess WMT17 ende training and evaluation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import gzip
import re

# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
import six
from six.moves import range
from six.moves import urllib
from six.moves import zip
import tensorflow.compat.v1 as tf

from brokenegg_transformer.utils.flags import core as flags_core
from brokenegg_transformer.utils import tokenizer
# pylint: enable=g-bad-import-order

# Data sources for training/evaluating the transformer translation model.
# If any of the training sources are changed, then either:
#   1) use the flag `--search` to find the best min count or
#   2) update the _TRAIN_DATA_MIN_COUNT constant.
# min_count is the minimum number of times a token must appear in the data
# before it is added to the vocabulary. "Best min count" refers to the value
# that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.
_WIKIMATRIX_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.%s.tsv.gz"

# <langpair> => <#samples>
_WIKIMATRIX_LANG_PAIR_SAMPLES = {
  # As of 2020 Jul 28
  'ar-de': 835734,
  'ar-el': 365898,
  'ar-en': 1968009,
  'ar-es': 829661,
  'ar-fr': 851422,
  'ar-ja': 669984,
  'ar-ko': 388887,
  'ar-ru': 821288,
  'ar-zh': 582415,
  'de-el': 770612,
  'de-en': 6227188,
  'de-es': 2550295,
  'de-fr': 3350816,
  'de-ja': 2271178,
  'de-ko': 913748,
  'de-ru': 2835270,
  'de-zh': 1358412,
  'el-en': 1407429,
  'el-es': 746432,
  'el-fr': 773559,
  'el-ja': 531379,
  'el-ko': 301106,
  'el-ru': 715980,
  'el-zh': 427862,
  'en-es': 6452177,
  'en-fr': 6562360,
  'en-ja': 3895992,
  'en-ko': 1345630,
  'en-ru': 5203872,
  'en-zh': 2595119,
  'es-fr': 2856402,
  'es-ja': 1802993,
  'es-ko': 854665,
  'es-ru': 2182862,
  'es-zh': 1214322,
  'fr-ja': 2010367,
  'fr-ko': 873398,
  'fr-ru': 2483459,
  'fr-zh': 1309915,
  'ja-ko': 968704,
  'ja-ru': 1950844,
  'ja-zh': 1325674,
  'ko-ru': 855551,
  'ko-zh': 486671,
  'ru-zh': 1264230,
}

# Strings to inclue in the generated files.
_PREFIX = "wikimatrix"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.

# Vocabulary constants
_SPM_TRAIN_FILE = "spm_train.txt"
_SPM_TRAIN_SAMPLES = 3000000
_VOCAB_SIZE = 32000
_VOCAB_SIZE_LARGE = 64000

# Number of files to split train and evaluation data
_TRAIN_SAMPLES_PER_SHARD = 45000
_EVAL_SAMPLES_PER_SHARD = 10000

_RANDOMIZE_INPUT_RATE = 0.3

###############################################################################
# Download and extraction functions
###############################################################################
def get_source_urls(raw_dir, url_template, lang_pair):
  url = url_template % (lang_pair,)
  return download_from_url(raw_dir, url)

def download_report_hook(count, block_size, total_size):
  """Report hook for download progress.

  Args:
    count: current block number
    block_size: block size
    total_size: total size
  """
  percent = int(count * block_size * 100 / total_size)
  print(six.ensure_str("\r%d%%" % percent) + " completed", end="\r")


def download_from_url(path, url):
  """Download content from a url.

  Args:
    path: string directory where file will be downloaded
    url: string url

  Returns:
    Full path to downloaded file
  """
  filename = six.ensure_str(url).split("/")[-1]
  filename = os.path.join(path, filename)
  if not tf.io.gfile.exists(filename):
    logging.info("Downloading from %s to %s." % (url, filename))
    inprogress_filepath = six.ensure_str(filename) + ".incomplete"
    inprogress_filepath, _ = urllib.request.urlretrieve(
        url, inprogress_filepath, reporthook=download_report_hook)
    # Print newline to clear the carriage return from the download progress.
    print()
    tf.gfile.Rename(inprogress_filepath, filename)
    return filename
  else:
    logging.info("Already downloaded: %s (at %s)." % (url, filename))
    return filename


###############################################################################
# Vocabulary
###############################################################################
def get_vocab_file(raw_dir, data_dir, vocab_file):
  return tokenizer.Subtokenizer(os.path.join(data_dir, vocab_file))


def make_spm_train_file(data_dir, lang_pairs, train_files):
  from collections import Counter
  spm_train_file = os.path.join(data_dir, _SPM_TRAIN_FILE)
  if os.path.exists(spm_train_file):
    logging.info("Already available: %s" % (spm_train_file,))
    return spm_train_file
  lang_count = Counter()
  for lang_pair in lang_pairs:
    lang1, lang2 = lang_pair.split('-')
    lang_count[lang1] += _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]
    lang_count[lang2] += _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]

  lang_rates = {
    lang: (_SPM_TRAIN_SAMPLES * 1.01) / (len(lang_count) * lang_count[lang])
    for lang in lang_count
  }

  with open(spm_train_file + '.incomplete', 'w') as fout:
    count = 0
    for lang_pair in lang_pairs:
      lang1, lang2 = lang_pair.split('-')
      train_file = train_files[lang_pair]
      with gzip.open(train_file, 'rt') as f:
        for line in f:
          parts = line.rstrip('\r\n').split('\t')
          if random.random() < lang_rates[lang1]:
            fout.write(parts[1] + '\n')
            count += 1
            if count % 500000 == 0:
              logging.info('%d lines written (%d%%)' % (count, count * 100 // _SPM_TRAIN_SAMPLES))
          if random.random() < lang_rates[lang2]:
            fout.write(parts[2] + '\n')
            count += 1
            if count % 500000 == 0:
              logging.info('%d lines written (%d%%)' % (count, count * 100 // _SPM_TRAIN_SAMPLES))

  os.rename(spm_train_file + '.incomplete', spm_train_file)

  return spm_train_file


def train_spm(spm_train_file, data_dir, vocab_file, vocab_size):
  import sentencepiece as spm
  model_prefix = os.path.join(data_dir, vocab_file)[:-len('.model')]
  spm.SentencePieceTrainer.train(
    f'--input={spm_train_file} --model_prefix={model_prefix} --vocab_size={vocab_size}')


###############################################################################
# Data preprocessing
###############################################################################
def encode_and_save_files(
    subtokenizer, data_dir, lang_pair, raw_files, total_train_shards, total_eval_shards, eval_ratio,
    prefix=_PREFIX, input_column=1, target_column=2, randomize_input=0.0):
  """Save data from files as encoded Examples in TFrecord format.

  Args:
    subtokenizer: Subtokenizer object that will be used to encode the strings.
    data_dir: The directory in which to write the examples
    raw_files: A tuple of (input, target) data files. Each line in the input and
      the corresponding line in target file will be saved in a tf.Example.
    tag: String that will be added onto the file names.
    total_shards: Number of files to divide the data into.

  Returns:
    List of all files produced.
  """
  # Create a file for each shard.
  train_filepaths = [shard_filename(data_dir, prefix, lang_pair, _TRAIN_TAG, n + 1, total_train_shards)
               for n in range(total_train_shards)]
  eval_filepaths = [shard_filename(data_dir, prefix, lang_pair, _EVAL_TAG, n + 1, total_eval_shards)
               for n in range(total_eval_shards)]
  filepaths = train_filepaths + eval_filepaths

  if all_exist(train_filepaths + eval_filepaths):
    logging.info("Files already exist.")
    return train_filepaths, eval_filepaths

  logging.info("Saving files.")

  # Write examples to each shard in round robin order.
  tmp_train_filepaths = [six.ensure_str(fname) + ".incomplete" for fname in train_filepaths]
  tmp_eval_filepaths = [six.ensure_str(fname) + ".incomplete" for fname in eval_filepaths]
  tmp_filepaths = tmp_train_filepaths + tmp_eval_filepaths
  train_writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_train_filepaths]
  eval_writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_eval_filepaths]
  train_counter, eval_counter = 0, 0
  for raw_file in raw_files:
    logging.info('Reading %s' % raw_file)

    with gzip.open(raw_file, 'rt') as f:
      for counter, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        if counter > 0 and counter % 100000 == 0:
          logging.info("\tSaving case %d of %s." % (counter, raw_file))

        input_text = parts[input_column]
        output_text = parts[target_column]
        encoded_input = subtokenizer.encode(input_text, add_eos=True)
        encoded_target = subtokenizer.encode(output_text, add_eos=True)
        for do_randomize in False, True:
          example = dict_to_example(
              {"inputs": encoded_input,
              "targets": encoded_target})
          if do_randomize or total_eval_shards == 0 or eval_counter >= eval_ratio * counter:
            shard = train_counter % total_train_shards
            train_writers[shard].write(example.SerializeToString())
            train_counter += 1
          else:
            shard = eval_counter % total_eval_shards
            eval_writers[shard].write(example.SerializeToString())
            eval_counter += 1
          if randomize_input > 0 and (randomize_input >= 1.0 or random.random() < randomize_input):
            input_text = _randomize_text(input_text)
            encoded_input = subtokenizer.encode(input_text, add_eos=True)
          else:
            break

  for writer in train_writers + eval_writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  logging.info("Saved %d Examples", counter + 1)
  return train_filepaths, eval_filepaths


def shard_filename(path, prefix, lang_pair, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%s-%.5d-of-%.5d" % (prefix, lang_pair, tag, shard_num, total_shards))


def shuffle_records(fname):
  """Shuffle records in a single file."""
  logging.info("Shuffling records in file %s" % fname)

  # Rename file prior to shuffling
  tmp_fname = six.ensure_str(fname) + ".unshuffled"
  tf.gfile.Rename(fname, tmp_fname)

  reader = tf.io.tf_record_iterator(tmp_fname)
  records = []
  for record in reader:
    records.append(record)
    if len(records) % 100000 == 0:
      logging.info("\tRead: %d", len(records))

  random.shuffle(records)

  # Write shuffled records to original file name
  with tf.python_io.TFRecordWriter(fname) as w:
    for count, record in enumerate(records):
      w.write(record)
      if count > 0 and count % 100000 == 0:
        logging.info("\tWriting record: %d" % count)

  tf.gfile.Remove(tmp_fname)


def dict_to_example(dictionary):
  """Converts a dictionary of string->int to a tf.Example."""
  features = {}
  for k, v in six.iteritems(dictionary):
    features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
  return tf.train.Example(features=tf.train.Features(feature=features))


def all_exist(filepaths):
  """Returns true if all files in the list exist."""
  for fname in filepaths:
    if not tf.gfile.Exists(fname):
      return False
  return True


def make_dir(path):
  if not tf.gfile.Exists(path):
    logging.info("Creating directory %s" % path)
    tf.gfile.MakeDirs(path)


def _swap_lang_pair(lang_pair):
  lang1, lang2 = lang_pair.split('-')
  return lang2 + '-' + lang1

def get_vocab_file_and_size():
  langs = set(
    lang
    for lang_pair in FLAGS.lang_pairs.split(',')
    for lang in lang_pair.split('-')
  )
  langs = sorted(langs)
  langs = '-'.join(langs)

  if FLAGS.lang_pairs:
    vocab_file = _PREFIX + "." + langs + ".spm32k.model"
    vocab_size = _VOCAB_SIZE
  else:
    vocab_file = _PREFIX + "_lang10.spm64k.model"
    vocab_size = _VOCAB_SIZE_LARGE

  return vocab_file, vocab_size


_NON_WORD_RX = re.compile('[^\s\w]+')

def _randomize_text(text):
  t = random.random()
  if t < 0.1:
    text = text.lower()
  elif t < 0.2:
    text = text.upper()
  elif t < 0.3:
    text = "*" + text
  elif t < 0.4:
    text = "." + text
  elif t < 0.5:
    text = "-" + text
  elif t < 0.6:
    text = _NON_WORD_RX.sub('', text)
  elif t < 0.7:
    # Swap two characters
    text = list(text)
    if len(text) >= 2:
      i = random.randint(0, len(text) - 2)
      text[i], text[i + 1] = text[i + 1], text[i]
    text = ''.join(text)
  elif t < 0.8:
    # Split the text into two and swap the two.
    if len(text) >= 2:
      i = random.randint(1, len(text) - 1)
      text = text[i:] + text[:i]
  elif t < 0.9:
    # Remove a span.
    if len(text) >= 2:
      l = random.randint(1, len(text) - 1)
      i = random.randint(0, len(text) - l)
      text = text[:i] + text[i+l:]
  else:
    # Shuffle words
    text = text.split()
    random.shuffle(text)
    text = ' '.join(text)
  return text.strip()


def main(unused_argv):
  """Obtain training and evaluation data for the Transformer model."""
  make_dir(FLAGS.raw_dir)
  make_dir(FLAGS.data_dir)
  if FLAGS.lang_pairs:
    lang_pairs = FLAGS.lang_pairs.split(',')
  else:
    lang_pairs = sorted(_WIKIMATRIX_LANG_PAIR_SAMPLES.keys())
    logging.info("--lang_pair is not given. Use:")
    for lang_pair in lang_pairs:
      logging.info("  %s" % lang_pair)

  # Download test_data
  logging.info("Step 1/5: Downloading test data")
  logging.info("Skipping downloading. We don't have test data.")
  #get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES)

  # Get paths of download/extracted training and evaluation files.
  logging.info("Step 2/5: Downloading data from source")
  train_files = {}
  if FLAGS.skip_wikimatrix:
    logging.info("No --skip_wikimatrix flag is given. Skipping.")
  else:
    for lang_pair in lang_pairs:
      train_file = get_source_urls(FLAGS.raw_dir, _WIKIMATRIX_URL_TEMPLATE, lang_pair)
      train_files[lang_pair] = train_file
      if not _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]:
        logging.info("Counting number of samples.")
        with gzip.open(train_file, 'rt') as f:
          n = len(f.readlines())
        _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair] = n
        with open('sample_count.txt', 'a') as f:
          f.write("  '%s': %d,\n" % (lang_pair, n))
        logging.info("%s: %d samples" % (lang_pair, n))

  # Create subtokenizer based on the training files.
  logging.info("Step 3/5: Creating sentencepiece and building vocabulary")
  vocab_file, vocab_size = get_vocab_file_and_size()
  if os.path.exists(os.path.join(FLAGS.data_dir, vocab_file)):
    logging.info("Already available: %s", (vocab_file,))
  else:
    spm_train_file = make_spm_train_file(FLAGS.data_dir, lang_pairs, train_files)
    train_spm(spm_train_file, FLAGS.data_dir, vocab_file, vocab_size)
  subtokenizer = get_vocab_file(FLAGS.raw_dir, FLAGS.data_dir, vocab_file)

  # Tokenize and save data as Examples in the TFRecord format.
  logging.info("Step 4/5: Preprocessing and saving data")
  if FLAGS.skip_wikimatrix:
    logging.info("No --skip_wikimatrix flag is given. Skipping.")
  else:
    for lang_pair in lang_pairs:
      train_file = train_files[lang_pair]
      num_samples = _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]
      train_shards = int(((1.0 + _RANDOMIZE_INPUT_RATE) * num_samples - _EVAL_SAMPLES_PER_SHARD) / _TRAIN_SAMPLES_PER_SHARD)
      assert train_shards > 0
      eval_shareds = 1
      eval_ratio = _EVAL_SAMPLES_PER_SHARD / num_samples
      for rev in False, True:
        input_column = 2 if rev else 1
        target_column = 1 if rev else 2
        new_lang_pair = _swap_lang_pair(lang_pair) if rev else lang_pair
        train_tfrecord_files, eval_tfrecord_files = encode_and_save_files(
            subtokenizer, FLAGS.data_dir, new_lang_pair, [train_file],
            train_shards, eval_shareds, eval_ratio,
            prefix=_PREFIX,
            input_column=input_column, target_column=target_column,
            randomize_input=_RANDOMIZE_INPUT_RATE)
        for fname in train_tfrecord_files:
          shuffle_records(fname)

  logging.info("Step 4/5: Preprocessing and saving extra data")
  if not FLAGS.extra_dir:
    logging.info("No --extra_dir flag is given. Skipping.")
  else:
    extra_files = [os.path.join(FLAGS.extra_dir, name) for name in tf.io.gfile.listdir(FLAGS.extra_dir)]
    train_shards = FLAGS.num_extra_samples // _TRAIN_SAMPLES_PER_SHARD
    eval_ratio = 0.0
    train_tfrecord_files, eval_tfrecord_files = encode_and_save_files(
        subtokenizer, FLAGS.data_dir, FLAGS.extra_lang_pair, extra_files,
        train_shards, 0, eval_ratio,
        prefix=FLAGS.extra_prefix,
        input_column=0, target_column=1)
    for fname in train_tfrecord_files:
      shuffle_records(fname)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", short_name="dd", default="/tmp/brokenegg_data",
      help=flags_core.help_wrap(
          "Directory for where the WikiMatrix dataset is saved."))
  flags.DEFINE_string(
      name="raw_dir", short_name="rd", default="/tmp/brokenegg_orig",
      help=flags_core.help_wrap(
          "Path where the raw data will be downloaded and extracted."))
  flags.DEFINE_string(
      name="lang_pairs", short_name="lp", default="",
      help=flags_core.help_wrap(
          "Language pairs to convert."))
  flags.DEFINE_string(
      name="extra_dir", short_name="ed", default="",
      help=flags_core.help_wrap(
          "Directory for where the extra dataset is found."))
  flags.DEFINE_string(
      name="extra_prefix", short_name="ep", default="extra",
      help=flags_core.help_wrap(
          "Prefix of extra data."))
  flags.DEFINE_string(
      name="extra_lang_pair", short_name="el", default="extra",
      help=flags_core.help_wrap(
          "Lang pair of extra data"))
  flags.DEFINE_integer(
      name="num_extra_samples", short_name="en", default=6000000, # 3,307,844
      help=flags_core.help_wrap(
          "Estimated number of extra samples."))
  flags.DEFINE_boolean(
      name="skip_wikimatrix", short_name="sw", default=False,
      help=flags_core.help_wrap(
          "Skip preprocessing wikimatrix"))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
