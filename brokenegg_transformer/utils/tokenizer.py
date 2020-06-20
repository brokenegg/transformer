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
"""Defines Subtokenizer class to encode and decode strings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.compat.v1 as tf
import tempfile

PAD = "<unk>"
PAD_ID = 0
SOS = "<s>"
SOS_ID = 1
EOS = "</s>"
EOS_ID = 2
RESERVED_TOKENS = [PAD, SOS, EOS]


class Subtokenizer(object):
  """Encodes and decodes strings to/from integer IDs."""

  def __init__(self, vocab_file, tmp_dir=None, reserved_tokens=None):
    """Initializes class, creating a vocab file if data_files is provided."""
    import sentencepiece as spm
    tf.logging.info("Initializing Subtokenizer from file %s." %
                              vocab_file)
    self.sp = spm.SentencePieceProcessor()

    if vocab_file.startswith('gs://'):
      with tempfile.TemporaryDirectory() as tmp_dir:
        for postfix in '.model', '.vocab':
          filename = vocab_file.rstrip('.model') + postfix
          if not tf.io.gfile.exists(filename):
            raise ValueError("File not found.")
          local_filename = os.path.join(tmp_dir, os.path.basename(filename))
          tf.io.gfile.copy(filename, local_filename, overwrite=False)
        local_filename = os.path.join(tmp_dir, os.path.basename(vocab_file))
        self.sp.load(local_filename)
    else:
      self.sp.load(vocab_file)

  @staticmethod
  def init_from_files(
      vocab_file, files, target_vocab_size, threshold, min_count=None,
      file_byte_limit=1e6, reserved_tokens=None, correct_strip=True):
    raise NotImplementedError()

  def encode(self, raw_string, add_eos=False):
    """Encodes a string into a list of int subtoken ids."""
    encoded = self.sp.encode_as_ids(raw_string)
    if add_eos:
      encoded.append([EOS_ID])
    return encoded

  def decode(self, subtokens):
    """Converts list of int subtokens ids into a string."""
    decoded = self.sp.decode_ids(subtokens)
    return decoded
