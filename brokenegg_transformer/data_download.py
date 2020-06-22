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
_WIKIMATRIX_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.%s-%s.tsv.gz"

# (<langpair>, <rate>), None for rate is equal to 1.0.
_WIKIMATRIX_LANG_PAIRS = [
  ('en-es', 0.5), # 6,452,177 pairs
  ('en-ja', None), # 3,895,992 pairs
  ('es-ja', None), # 1,802,993 pairs
]

# Strings to inclue in the generated files.
_PREFIX = "brokenegg"
_TRAIN_TAG = "train"
_EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
# evaluation datasets are tagged as "dev" for development.

# Vocabulary constants
_VOCAB_FILE = _PREFIX + ".en-es-ja.spm64k.model"

# Number of files to split train and evaluation data
_TRAIN_SHARDS = 40
_EVAL_SHARDS = 1
_TRAIN_EVAL_RATIO = 10


###############################################################################
# Download and extraction functions
###############################################################################
def get_source_urls(raw_dir, url_template, lang_pairs):
  res = []
  for lang_pair, rate in lang_pairs:
    lang1, lang2 = lang_pair.split('-')
    url = _WIKIMATRIX_URL_TEMPLATE % (lang1, lang2)
    filename = download_from_url(raw_dir, url)
    res.append((lang1, lang2, rate, filename))
  return res

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


def get_vocab_file(raw_dir, data_dir, vocab_file):
  return tokenizer.Subtokenizer(os.path.join(data_dir, vocab_file))


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
    subtokenizer, data_dir, raw_files, total_train_shards, total_eval_shards, lang_map, extra_files=[], train_eval_ratio=100):
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
  train_filepaths = [shard_filename(data_dir, _TRAIN_TAG, n + 1, total_train_shards)
               for n in range(total_train_shards)]
  eval_filepaths = [shard_filename(data_dir, _EVAL_TAG, n + 1, total_eval_shards)
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
  writers = train_writers * train_eval_ratio + eval_writers
  total_shards = len(writers)
  counter, shard = 0, 0
  for lang1, lang2, rate, raw_file in raw_files:
    logging.info('Reading %s' % raw_file)
    lang1_sos = lang_map.get(lang1, tokenizer.SOS_ID)
    lang2_sos = lang_map.get(lang2, tokenizer.SOS_ID)

    with gzip.open(raw_file, 'rt') as f:
      for counter, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        if counter > 0 and counter % 100000 == 0:
          logging.info("\tSaving case %d of %s." % (counter, raw_file))
          break

        encoded_input = subtokenizer.encode(parts[1])
        encoded_target = subtokenizer.encode(parts[2])
        if rate is None or random.random() < rate:
          example = dict_to_example(
              {"inputs": encoded_input,
              "targets": [lang2_sos] + encoded_target})
          writers[shard].write(example.SerializeToString())

          shard = (shard + 1) % total_shards

        # Rev
        if rate is None or random.random() < rate:
          example = dict_to_example(
              {"inputs": encoded_target,
              "targets": [lang1_sos] + encoded_input})
          writers[shard].write(example.SerializeToString())

          shard = (shard + 1) % total_shards

  shard = 0
  for raw_file in extra_files:
    with gzip.open(raw_file, 'rt') as f:
      for counter, line in enumerate(f):
        parts = line.rstrip('\r\n').split('\t')
        if counter > 0 and counter % 100000 == 0:
          logging.info("\tSaving case %d of %s." % (counter, raw_file))
          break

        encoded_input = subtokenizer.encode(parts[0])
        encoded_target = subtokenizer.encode(parts[1])
        example = dict_to_example(
            {"inputs": encoded_input,
            "targets": [tokenizer.SOS_ID] + encoded_target})
        train_writers[shard].write(example.SerializeToString())

        shard = (shard + 1) % len(train_writers)

  for writer in train_writers + eval_writers:
    writer.close()

  for tmp_name, final_name in zip(tmp_filepaths, filepaths):
    tf.gfile.Rename(tmp_name, final_name)

  logging.info("Saved %d Examples", counter + 1)
  return train_filepaths, eval_filepaths


def shard_filename(path, tag, shard_num, total_shards):
  """Create filename for data shard."""
  return os.path.join(
      path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))


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

  # Download test_data
  logging.info("Step 1/4: Downloading test data")
  logging.info("Skipping downloading. We don't have test data.")
  #get_raw_files(FLAGS.data_dir, _TEST_DATA_SOURCES)

  # Get paths of download/extracted training and evaluation files.
  logging.info("Step 2/4: Downloading data from source")
  train_files = get_source_urls(FLAGS.raw_dir, _WIKIMATRIX_URL_TEMPLATE, _WIKIMATRIX_LANG_PAIRS)

  # Create subtokenizer based on the training files.
  logging.info("Step 3/4: Creating sentencepiece and building vocabulary")
  subtokenizer = get_vocab_file(FLAGS.raw_dir, FLAGS.data_dir, _VOCAB_FILE)

  # Tokenize and save data as Examples in the TFRecord format.
  logging.info("Step 4/4: Preprocessing and saving data")
  lang_map = get_lang_map(subtokenizer, _WIKIMATRIX_LANG_PAIRS)
  extra_files = [os.path.join(FLAGS.extra_dir, name) for name in tf.io.gfile.listdir(FLAGS.extra_dir)]
  train_tfrecord_files, eval_tfrecord_files = encode_and_save_files(
      subtokenizer, FLAGS.data_dir, train_files,
      _TRAIN_SHARDS, _EVAL_SHARDS, lang_map, extra_files=extra_files, train_eval_ratio=_TRAIN_EVAL_RATIO)

  for fname in train_tfrecord_files:
    shuffle_records(fname)


def define_data_download_flags():
  """Add flags specifying data download arguments."""
  flags.DEFINE_string(
      name="data_dir", short_name="dd", default="/tmp/brokenegg_transformer",
      help=flags_core.help_wrap(
          "Directory for where the WikiMatrix dataset is saved."))
  flags.DEFINE_string(
      name="raw_dir", short_name="rd", default="/tmp/brokenegg_transformer",
      help=flags_core.help_wrap(
          "Path where the raw data will be downloaded and extracted."))
  flags.DEFINE_string(
      name="extra_dir", short_name="ed", default="/tmp/brokenegg_transformer/extra",
      help=flags_core.help_wrap(
          "Directory for where the extra dataset is found."))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_data_download_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)
