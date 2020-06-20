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
"""Test Subtokenizer and string helper methods."""

from brokenegg_transformer.utils import tokenizer

import tensorflow as tf

class SubtokenizerTest(tf.test.TestCase):

  def test_simple(self):
    path = 'gs://brokenegg/data/brokenegg/spm.en-es-ja.spm64k.model'
    path = '/tmp/brokenegg_transformer/spm.en-es-ja.spm64k.model'
    subtokenizer = tokenizer.Subtokenizer(path)
    text = "Hello, world! こんにちはです。"
    encoded = subtokenizer.encode(text)
    print(encoded)
    decoded = subtokenizer.decode(encoded)
    self.assertEqual(decoded, text)

if __name__ == "__main__":
  tf.test.main()
