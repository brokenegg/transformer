# Copyright 2020 Katsuya Iida.

from brokenegg_transformer.utils import tokenizer

import os
import tensorflow.compat.v1 as tf

data_dir = '/tmp/brokenegg_transformer'
dataset = tf.data.TFRecordDataset(os.path.join(data_dir, 'brokenegg-train-00030-of-00030'))
feature_description = {
    'inputs': tf.VarLenFeature(dtype=tf.int64),
    'targets': tf.VarLenFeature(dtype=tf.int64),
}
#feature_description = {
#    'inputs': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
#    'targets': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
#}
subtokenizer = tokenizer.Subtokenizer(os.path.join(data_dir, 'brokenegg.en-es-ja.spm64k.model'))
for count, raw_record in enumerate(dataset):
    #print(raw_record)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(example)
    
    example = tf.io.parse_single_example(raw_record, feature_description)
    encoded_inputs = tf.sparse.to_dense(example['inputs']).numpy().tolist()
    encoded_targets = tf.sparse.to_dense(example['targets']).numpy().tolist()
    print('LANG: %d' % encoded_targets[0])
    print('SRC: %s' % subtokenizer.decode(encoded_inputs))
    print('TGT: %s' % subtokenizer.decode(encoded_targets[1:]))
    if count > 10:
        break