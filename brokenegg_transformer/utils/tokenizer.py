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

import collections
import re
import sys
import unicodedata

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

PAD = "<pad>"
PAD_ID = 0
EOS = "<EOS>"
EOS_ID = 1
RESERVED_TOKENS = [PAD, EOS]


class Subtokenizer(object):
  """Encodes and decodes strings to/from integer IDs."""

  def __init__(self, vocab_file, reserved_tokens=None):
    """Initializes class, creating a vocab file if data_files is provided."""
    tf.compat.v1.logging.info("Initializing Subtokenizer from file %s." %
                              vocab_file)

  @staticmethod
  def init_from_files(
      vocab_file, files, target_vocab_size, threshold, min_count=None,
      file_byte_limit=1e6, reserved_tokens=None, correct_strip=True):
    raise NotImplementedError()

  def encode(self, raw_string, add_eos=False):
    """Encodes a string into a list of int subtoken ids."""
    raise NotImplementedError()

  def decode(self, subtokens):
    """Converts list of int subtokens ids into a string."""
    raise NotImplementedError()
