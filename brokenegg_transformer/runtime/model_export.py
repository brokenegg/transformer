# Copyright 2020 Katsuya Iida.

from brokenegg_transformer.runtime.transformer import load_model_as_function

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np
import os

PAD_ID = 0
EOS_ID = 2

flags.DEFINE_string(
    name="model_dir", short_name="md", default="/tmp",
    help="The location of the model data.")
flags.DEFINE_string(
    name="weight_file", short_name="wf", default="brokenegg-20200711.npz",
    help="NumPy weight file.")
flags.DEFINE_string(
    name="format", short_name="e", default="tflite",
    help="Model format.")


def export_numpy(checkpoint_path, weight_file):
  import re
  from brokenegg_transformer import model_params
  from brokenegg_transformer import transformer

  def clean_variable_name(name):
      name = name.replace('transformer_v2_1/Transformer/', '')
      name = name.replace('transformer_v2/Transformer/', '')
      name = name.replace(':0', '')
      name = re.sub(r'(embedding_shared_weights|pre_post_processing_wrapper|encoder_stack|decoder_stack|self_attention|layer_normalization|feed_forward_network|attention)_[0-9]+', r'\1', name)
      return name

  params = model_params.BASE_PARAMS.copy()
  params["dtype"] = tf.float32
  model = transformer.create_model(params, is_train=True, has_initial_ids=False)

  ckpt_path = tf.train.latest_checkpoint(checkpoint_path)
  print('Restoring from %s' % ckpt_path)
  #model.load_weights(ckpt_path)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(ckpt_path).assert_existing_objects_matched().expect_partial()
  
  arr = {
    clean_variable_name(v.name): v.numpy()
    for v in model.trainable_variables
  }

  np.savez(weight_file, **arr)

def export_tflite_tf22(weight_file='examples/brokenegg.npz', model_file='brokenegg_tf22.tflite', max_len=10):
  assert tf.__version__.split('.')[0] == '2'
  assert tf.__version__.split('.')[1] == '2'
  func = load_model_as_function(weight_file, max_len=max_len)
  inputs_data = tf.TensorSpec(shape=[None, max_len], dtype=tf.int64)
  targets_data = tf.TensorSpec(shape=[None, max_len], dtype=tf.int64)
  cfunc = func.get_concrete_function(inputs_data, targets_data)
  converter = tf.lite.TFLiteConverter.from_concrete_functions([cfunc])
  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  #converter.target_spec.supported_types = [tf.float16]
  res = converter.convert()
  with tf.io.gfile.GFile(model_file, 'wb') as f:
    f.write(res)

def export_tflite_tf23(weight_file='examples/brokenegg.npz', model_file='brokenegg_tf23_fp16.tflite'):
  assert tf.__version__.split('.')[0] == '2'
  assert tf.__version__.split('.')[1] == '3'
  func = load_model_as_function(weight_file)
  inputs_data = tf.TensorSpec(shape=[None, None], dtype=tf.int64)
  targets_data = tf.TensorSpec(shape=[None, None], dtype=tf.int64)
  cfunc = func.get_concrete_function(inputs_data, targets_data)
  converter = tf.lite.TFLiteConverter.from_concrete_functions([cfunc])
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  res = converter.convert()
  with tf.io.gfile.GFile(model_file, 'wb') as f:
    f.write(res)

def test_tflite_tf22(model_file='brokenegg.tflite',
    vocab_file='examples/model_base_20200623/brokenegg.en-es-ja.spm64k.model',
    max_len=10):
  import sentencepiece as spm
  sp = spm.SentencePieceProcessor()
  sp.load(vocab_file)
  interpreter = tf.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_text = 'I went to school, today.'
  print('IN: %s' % input_text)
  x = sp.encode_as_ids(input_text) + [EOS_ID] + [PAD_ID] * max_len
  inputs_data = np.array([x[:max_len]], dtype=np.int64)
  targets_data = np.array([[1] + [PAD_ID] * (max_len - 1)], np.int64)

  for i in range(10):
    interpreter.set_tensor(input_details[0]['index'], inputs_data)
    interpreter.set_tensor(input_details[1]['index'], targets_data)
    interpreter.invoke()
    outputs_data = interpreter.get_tensor(output_details[0]['index'])
    predicts = outputs_data[:, i, :].argmax(axis=-1)
    print(predicts[0])
    if predicts[0] == EOS_ID:
      break
    targets_data = np.concatenate([
        targets_data[:, :i + 1],
        [predicts],
        targets_data[:, i + 2:]],
      axis=1)
    target_text = sp.decode_ids(targets_data[0, 1:i+2].tolist())
    print('OUT: %s' % target_text)

def test_tflite_tf23(model_file='brokenegg_tf23_fp16.tflite',
    vocab_file='examples/model_base_20200623/brokenegg.en-es-ja.spm64k.model',
    max_len=10):
  import sentencepiece as spm
  sp = spm.SentencePieceProcessor()
  sp.load(vocab_file)
  interpreter = tf.lite.Interpreter(model_path=model_file)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_text = 'I went to school, today.'
  print('IN: %s' % input_text)
  inputs_data = np.array([sp.encode_as_ids(input_text) + [EOS_ID]], dtype=np.int64)
  targets_tokens = [64002]

  for i in range(max_len):
    if i == 0:
      targets_len = 10
      interpreter.resize_tensor_input(0, inputs_data.shape)
      interpreter.resize_tensor_input(1, [1, targets_len])
      interpreter.allocate_tensors()

    targets_data = np.array([(targets_tokens + [PAD_ID] * targets_len)[:targets_len]], dtype=np.int64)
    interpreter.set_tensor(input_details[0]['index'], inputs_data)
    interpreter.set_tensor(input_details[1]['index'], targets_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predict = int(output_data[0, i])
    if predict == 2:
      break
    targets_tokens.append(predict)
    target_text = sp.decode_ids(targets_tokens[1:])
    print('OUT: %s' % target_text)

def export_saved_model(weight_file='examples/brokenegg.npz', output_dir='saved_model'):
  from brokenegg_transformer.runtime import transformer
  model = transformer.load_model_as_model(weight_file)
  model.save(output_dir)

def export_onnx(weight_file, output_file='brokenegg-20200711.onnx'):

  import tensorflow as tf
  import tf2onnx

  with tf.Session(graph=graph) as sess:
    onnx_graph = tf2onnx.tensorflow_to_onnx(sess.graph,
      input_names=["inputs:0", "targets:0"], output_names=["logits:0"])
    model_proto = onnx_graph.make_model("test")
    with open(output_file, "wb") as f:
        f.write(model_proto.SerializeToString())

def model_test():
  from brokenegg_transformer import model_params
  from brokenegg_transformer import transformer

  params = model_params.BASE_PARAMS.copy()
  params["dtype"] = tf.float32
  if True:
    inputs = tf.keras.layers.Input((None,), dtype="int64", name="inputs")
    targets = tf.keras.layers.Input((None,), dtype="int64", name="targets")
    internal_model = transformer.Transformer(params, name="transformer_v2")
    logits = internal_model([inputs, targets], training=True)
    model = tf.keras.Model([inputs, targets], logits)
  else:
    model = transformer.create_model(params, is_train=True, has_initial_ids=True)

  ckpt_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
  print('Restoring from %s' % ckpt_path)
  #model.load_weights(ckpt_path)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.restore(ckpt_path).assert_existing_objects_matched().expect_partial()
  print('*************')
  tf.saved_model.save(model, 'saved_model')

def export_tvm(weight_file='examples/brokenegg.npz', model_path=None):
  import tvm.relay.testing.tf as tf_testing
  from tvm import relay
  import tvm

  from brokenegg_transformer.runtime import transformer

  graph, [inputs, targets], _ = transformer.load_model(weight_file, as_graph=True)
  with tf.Session(graph=graph) as sess:
    graph_def = tf.graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), ["logits"]
    )
  if False:

    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    with tf.Session(graph=graph) as sess:
      graph_def = tf_testing.AddShapesToGraphDef(sess, 'logits')

  layout = None
  target = 'llvm'
  target_host = 'llvm'

  shape_dict = {
    'inputs': [None, None],
    'targets': [None, None]
    }
  dtype_dict = {
    'inputs': "int64",
    'targets': "int64",
  }
  with tf.Session(graph=graph) as sess:
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                              layout=layout,
                                              shape=shape_dict,
                                              outputs=["logits"])
                                             
  with tvm.transform.PassContext(opt_level=3):
    graph, lib, params = relay.build(mod,
                                    target=target,
                                    target_host=target_host,
                                    params=params)

def main(_):
  flags_obj = flags.FLAGS
  if flags_obj.format == 'numpy':
    export_numpy(flags_obj.model_dir, flags_obj.weight_file)
  elif flags_obj.format == 'tflite':
    export_tflite_tf23()
  elif flags_obj.format == 'tflite_test':
    test_tflite_tf23()
  elif flags_obj.format == 'tflite_tf22':
    export_tflite_tf22()
  elif flags_obj.format == 'tflite_tf22_test':
    test_tflite_tf22()
  elif flags_obj.format == 'saved_model':
    export_saved_model(weight_file=flags_obj.weight_file)
  elif flags_obj.format == 'onnx':
    export_onnx(weight_file=flags_obj.weight_file)
  else:
    raise ValueError()

if __name__ == '__main__':
  app.run(main)
