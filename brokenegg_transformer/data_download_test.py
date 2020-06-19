import tensorflow.compat.v1 as tf

dataset = tf.data.TFRecordDataset('/tmp/brokenegg_transformer/brokenegg-train-00030-of-00030')
feature_description = {
    'inputs': tf.VarLenFeature(dtype=tf.int64),
    'targets': tf.VarLenFeature(dtype=tf.int64),
}
#feature_description = {
#    'inputs': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
#    'targets': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
#}
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('/tmp/brokenegg_transformer/spm.en-es-ja.spm64k.model')
for count, raw_record in enumerate(dataset):
    #print(raw_record)
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    #print(example)
    
    example = tf.io.parse_single_example(raw_record, feature_description)
    print('SRC: ' + sp.decode_ids(tf.sparse.to_dense(example['inputs']).numpy().tolist()))
    print('TGT: ' + sp.decode_ids(tf.sparse.to_dense(example['targets']).numpy().tolist()[1:]))
    if count > 10:
        break