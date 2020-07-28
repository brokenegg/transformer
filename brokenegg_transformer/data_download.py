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
  'ar-de': None, # 99258
  'ar-en': None, # 999762
  'ar-es': None, # 174557
  'ar-fr': None, # 163549
  'ar-gl': None, # 50528
  'ar-ja': None, # 83059
  'ar-ko': None, # 48869
  'ar-ru': None, # 125312
  'ar-zh': None, # 86236
  'de-en': None, # 1573437
  'de-es': None, # 418724
  'de-fr': None, # 626166
  'de-gl': None, # 80842
  'de-ja': None, # 217547
  'de-ko': None, # 82280
  'de-ru': None, # 368206
  'de-zh': None, # 134077
  'en-es': None, # 3377911
  'en-fr': None, # 2757883
  'en-gl': None, # 446151
  'en-ja': None, # 851706
  'en-ko': None, # 306900
  'en-ru': None, # 1661908
  'en-zh': None, # 786511
  'es-fr': None, # 905760
  'es-gl': None, # 610824
  'es-ja': None, # 219260
  'es-ko': None, # 108385
  'es-ru': None, # 393314
  'es-zh': None, # 174315
  'fr-gl': None, # 154872
  'fr-ja': None, # 214852
  'fr-ko': None, # 89109
  'fr-ru': None, # 410005
  'fr-zh': None, # 157013
  'gl-ja': None, # 50922
  'gl-ko': None, # 28478
  'gl-ru': None, # 84460
  'gl-zh': None, # 46609
  'ja-ko': None, # 222118
  'ja-ru': None, # 196556
  'ja-zh': None, # 267409
  'ko-ru': None, # 89951
  'ko-zh': None, # 57932
  'ru-zh': None, # 148733
}

# Strings to inclue in the generated files.
_PREFIX = "wikimatrix"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.

# Vocabulary constants
_SPM_TRAIN_FILE = _PREFIX + "spm_train.en-es-ja.txt"
_SPM_TRAIN_SAMPLES = 3000000
_VOCAB_FILE = _PREFIX + ".en-es-ja.spm64k.model"
_VOCAB_SIZE = 64000

# Number of files to split train and evaluation data
_TRAIN_SAMPLES_PER_SHARD = 45000
_EVAL_SAMPLES_PER_SHARD = 10000


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


def train_spm(spm_train_file, data_dir, vocab_file):
  import sentencepiece as spm
  model_prefix = os.path.join(data_dir, vocab_file)[:-len('.model')]
  spm.SentencePieceTrainer.train(
    f'--input={spm_train_file} --model_prefix={model_prefix} --vocab_size={_VOCAB_SIZE}')


###############################################################################
# Data preprocessing
###############################################################################
def all_langs(lang_pairs):
  langs = set()
  for langpair, _ in lang_pairs:
    inputs_lang, targets_lang = langpair.split('-')
    langs.add(inputs_lang)
    langs.add(targets_lang)
  return sorted(list(langs))


def get_lang_map(subtokenizer, lang_pairs):
  langs = all_langs(lang_pairs)
  offset = subtokenizer.vocab_size
  return {v: offset + k for k, v in enumerate(langs)}


def encode_and_save_files(
    subtokenizer, data_dir, lang_pair, raw_files, total_train_shards, total_eval_shards, eval_ratio,
    input_column=1, target_column=2):
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
  train_filepaths = [shard_filename(data_dir, lang_pair, _TRAIN_TAG, n + 1, total_train_shards)
               for n in range(total_train_shards)]
  eval_filepaths = [shard_filename(data_dir, lang_pair, _EVAL_TAG, n + 1, total_eval_shards)
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

        encoded_input = subtokenizer.encode(parts[input_column], add_eos=True)
        encoded_target = subtokenizer.encode(parts[target_column], add_eos=True)
        example = dict_to_example(
            {"inputs": encoded_input,
            "targets": encoded_target})
        if total_eval_shards == 0 or eval_counter >= eval_ratio * counter:
          shard = train_counter % total_train_shards
          train_writers[shard].write(example.SerializeToString())
          train_counter += 1
        else:
          shard = eval_counter % total_eval_shards
          eval_writers[shard].write(example.SerializeToString())
          eval_counter += 1

  for writer in train_writers + eval_writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  logging.info("Saved %d Examples", counter + 1)
  return train_filepaths, eval_filepaths


def shard_filename(path, lang_pair, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%s-%.5d-of-%.5d" % (_PREFIX, lang_pair, tag, shard_num, total_shards))


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


def main(unused_argv):
  """Obtain training and evaluation data for the Transformer model."""
  make_dir(FLAGS.raw_dir)
  make_dir(FLAGS.data_dir)
  if FLAGS.lang_pairs:
    lang_pairs = FLAGS.lang_pairs.split(',')
  else:
    lang_pairs = sorted(_WIKIMATRIX_LANG_PAIR_SAMPLES.keys())
    logging.info("Language pair is not given. Use:")
    for lang_pair in lang_pairs:
      logging.info("  %s" % lang_pair)

  # Download test_data
  logging.info("Step 1/5: Downloading test data")
  logging.info("Skipping downloading. We don't have test data.")
  #get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES)

  # Get paths of download/extracted training and evaluation files.
  logging.info("Step 2/5: Downloading data from source")
  train_files = {}
  for lang_pair in lang_pairs:
    train_file = get_source_urls(FLAGS.raw_dir, _WIKIMATRIX_URL_TEMPLATE, lang_pair)
    train_files[lang_pair] = train_file
    if not _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]:
      logging.info("Counting number of samples.")
      with gzip.open(train_file, 'rt') as f:
        n = len(f.readlines())
      _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair] = n
      with open('sample_count.txt', 'wa') as f:
        f.write("  '%s': %d,\n" % (lang_pair, n))
      logging.info("%s: %d samples" % (lang_pair, n))

  # Create subtokenizer based on the training files.
  logging.info("Step 3/5: Creating sentencepiece and building vocabulary")
  if os.path.exists(os.path.join(FLAGS.data_dir, _VOCAB_FILE)):
    logging.info("Already available: %s", (_VOCAB_FILE,))
  else:
    spm_train_file = make_spm_train_file(FLAGS.data_dir, lang_pairs, train_files)
    train_spm(spm_train_file, FLAGS.data_dir, _VOCAB_FILE)
  subtokenizer = get_vocab_file(FLAGS.raw_dir, FLAGS.data_dir, _VOCAB_FILE)

  # Tokenize and save data as Examples in the TFRecord format.
  logging.info("Step 4/5: Preprocessing and saving data")
  for lang_pair in lang_pairs:
    train_file = train_files[lang_pair]
    num_samples = _WIKIMATRIX_LANG_PAIR_SAMPLES[lang_pair]
    train_shards = int((num_samples - _EVAL_SAMPLES_PER_SHARD) / _TRAIN_SAMPLES_PER_SHARD)
    eval_shareds = 1
    eval_ratio = _EVAL_SAMPLES_PER_SHARD / num_samples
    train_tfrecord_files, eval_tfrecord_files = encode_and_save_files(
        subtokenizer, FLAGS.data_dir, lang_pair, [train_file],
        train_shards, eval_shareds, eval_ratio)
    for fname in train_tfrecord_files:
      shuffle_records(fname)

  logging.info("Step 4/5: Preprocessing and saving extra data")
  extra_files = [os.path.join(FLAGS.extra_dir, name) for name in tf.io.gfile.listdir(FLAGS.extra_dir)]
  train_shards = FLAGS.num_extra_samples // _TRAIN_SAMPLES_PER_SHARD
  eval_ratio = 0.0
  train_tfrecord_files, eval_tfrecord_files = encode_and_save_files(
      subtokenizer, FLAGS.data_dir, FLAGS.extra_prefix, extra_files,
      train_shards, 0, eval_ratio,
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
      name="extra_dir", short_name="ed", default="/tmp/brokenegg_orig/extra",
      help=flags_core.help_wrap(
          "Directory for where the extra dataset is found."))
  flags.DEFINE_string(
      name="extra_prefix", short_name="ep", default="extra",
      help=flags_core.help_wrap(
          "Prefix of extra data."))
  flags.DEFINE_integer(
      name="num_extra_samples", short_name="en", default=6000000, # 6,996,128 
      help=flags_core.help_wrap(
          "Estimated number of extra samples."))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
