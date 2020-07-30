# Copyright 2020 Katsuya Iida.

def test():
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
  #  'inputs': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
  #  'targets': tf.FixedLenFeature(shape=[1, None], dtype=tf.int64),
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

def main():
  import re
  import urllib
  from collections import Counter
  import os

  supported_langs = {
    'en', 'es', 'fr', 'ru', 'de', 'ja', 'ar', 'zh', 'el', 'ko'
  }

  rx = re.compile(r'[^.]+\.([^.]+)\.tsv\s+([0-9]+)')
  url = 'https://github.com/facebookresearch/LASER/raw/master/tasks/WikiMatrix/list_of_bitexts.txt'
  file = os.path.basename(url)
  if not os.path.exists(file):
    file, _ = urllib.request.urlretrieve(url, file)
  c = Counter()
  with open(file) as f:
    for line in f:
      m = rx.match(line)
      if m:
        lang_pair, n = m[1], int(m[2])
        #print(lang_pair, n)
        lang1, lang2 = lang_pair.split('-')
        c[lang1] += n
        c[lang2] += n

  sorted_langs = sorted(c.items(), key=lambda x: x[1], reverse=True)
  for lang, n in sorted_langs[:50]:
    pass
    #print(lang, n)

  with open(file) as f:
    for line in f:
      m = rx.match(line)
      if m:
        lang_pair, n = m[1], int(m[2])
        #print(lang_pair, n)
        lang1, lang2 = lang_pair.split('-')
        if lang1 in supported_langs and lang2 in supported_langs:
          print("'%s-%s': None, # %d" % (lang1, lang2, n))

main()