import tensorflow as tf
from brokenegg_transformer.runtime.transformer import load_model_as_model
import sentencepiece as spm

PAD_ID = 0
EOS_ID = 2

def test(model_file='examples/brokenegg.npz', vocab_file='examples/model_base_20200623/brokenegg.en-es-ja.spm64k.model'):
  sp = spm.SentencePieceProcessor()
  sp.load(vocab_file)
  model = load_model_as_model(model_file)

  input_text = "I went to the school."
  inputs_data = tf.constant([sp.encode_as_ids(input_text) + [EOS_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID, PAD_ID]], dtype=tf.int64)
  targets_data = tf.constant([[64002, EOS_ID]], dtype=tf.int64)
  while True:
    logits = model(inputs_data, targets_data, training=False)
    predicts = tf.argmax(logits[:, -1, :], axis=-1)
    print(predicts)
    if targets_data.shape[1] > 20 or predicts[0] == EOS_ID:
      break
    targets_data = tf.concat([
      targets_data[:, :-1],
      predicts[None, None, -1],
      targets_data[:, -1:],
    ], axis=1)
    target_text = sp.decode_ids(targets_data[0, 1:].numpy().tolist())
    print(target_text)

test()