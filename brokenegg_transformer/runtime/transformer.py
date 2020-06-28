# Copyright Katsuya Iida.

import tensorflow as tf
import numpy as np
import math

# Very low numbers to represent -infinity. We do not actually use -Inf, since we
# want to be able to multiply these values by zero to get zero. (-Inf * 0 = NaN)
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min

# Variables

variable_space = ''
trainable_variables = {}

class _VariableScope:
  def __init__(self, name):
    self.name = name
  def __enter__(self):
    global variable_space
    self.parent = variable_space
    if variable_space:
      variable_space += '/' + self.name
    else:
      variable_space = self.name
  def __exit__(self, a, b, c):
    global variable_space
    variable_space = self.parent

def variable_scope(name):
  return _VariableScope(name)

def get_variable(name):
  return trainable_variables[variable_space + '/' + name]

def set_variable(name, value):
  trainable_variables[variable_space + '/' + name] = value

def set_variables(vars):
  global trainable_variables
  trainable_variables = {
    k: tf.constant(v)
    for k, v in vars.items()
  }

def list_variables():
  return [
    (k, v.shape)
    for k, v in trainable_variables.items()
  ]

# Misc

def einsum(subscripts, a, b):
  if subscripts == 'abc,cde->abde':
    return tf.tensordot(a, b, axes=[[2], [0]])
  elif subscripts ==  "BTNH,BFNH->BNFT":
    a = tf.transpose(a, [0, 2, 3, 1])
    b = tf.transpose(b, [0, 2, 1, 3])
    return tf.matmul(b, a)
  elif subscripts == "BNFT,BTNH->BFNH":
    #a = tf.transpose(a, [0, 2, 3, 1])
    b = tf.transpose(b, [0, 2, 1, 3])
    c = tf.matmul(a, b)
    return tf.transpose(c, [0, 2, 1, 3])
  elif subscripts == 'abcd,cde->abe':
    return tf.tensordot(a, b, axes=[[2,3], [0,1]])
  elif subscripts == 'abc,cd->abd':
    return tf.tensordot(a, b, axes=[[2], [0]])
  else:
    raise Exception(subscripts)
    #return tf.einsum(subscripts, a, b)

def layer_norm(inputs, epsilon=0.001):
  with variable_scope('layer_normalization'):
    mean = tf.expand_dims(tf.reduce_mean(inputs, axis=-1), axis=-1) 
    var = tf.expand_dims(tf.reduce_mean((inputs - mean) ** 2, axis=-1), axis=-1)
    norm_inputs = (inputs - mean) / tf.sqrt(var + epsilon)
    gamma = get_variable('gamma')
    beta = get_variable('beta')
    return gamma * norm_inputs + beta

def dense_layer(name, inputs, subscripts='abc,cde->abde', use_bias=True, activation=None):
  with variable_scope(name):
    kernel = get_variable('kernel')
    y = einsum(subscripts, inputs, kernel)
    if use_bias:
      bias = get_variable('bias')
      y += bias
    if activation is not None:
      y = activation(y)
    return y

# Transformer layers

def embedding_softmax_layer(inputs, hidden_size=512, mode="embedding"):
  with variable_scope('embedding_shared_weights'):
    with variable_scope('embedding_and_softmax'):
      shared_weights = get_variable('weights')
      if mode == "embedding":
        embedded_inputs = tf.gather(shared_weights, inputs)
        embedded_inputs *= hidden_size ** 0.5
        return embedded_inputs
      elif mode == "linear":
        vocab_size = tf.shape(shared_weights)[0]
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        x = tf.reshape(inputs, [-1, hidden_size])
        logits = tf.matmul(x, shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, vocab_size])

def get_decoder_self_attention_bias(length, dtype=tf.float32):
  """Calculate bias for decoder that maintains model's autoregressive property.

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.

  Args:
    length: int length of sequences in batch.
    dtype: The dtype of the return value.

  Returns:
    float tensor of shape [1, 1, length, length]
  """
  neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                     -1, 0)
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = neg_inf * (1.0 - valid_locs)
  return decoder_bias

def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.

  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  return signal

def pre_post_processing_wrapper(layer, x, *args):
  with variable_scope("pre_post_processing_wrapper"):
    y = layer_norm(x, epsilon=1e-6)
    y = layer(y, *args)
    return x + y

def self_attention_layer(query_input, bias, name="self_attention", **args):
  return attention_layer(query_input, query_input, bias, name=name, **args)

def attention_layer(query_input, source_input, bias, name="attention", hidden_size=512, num_heads=8):
  with variable_scope(name):
    query = dense_layer('query', query_input, use_bias=False)
    key = dense_layer('key', source_input, use_bias=False)
    value = dense_layer('value', source_input, use_bias=False)

    depth = (hidden_size // num_heads)
    query *= depth ** -0.5

    logits = einsum("BTNH,BFNH->BNFT", key, query)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits)
    attention_output = einsum("BNFT,BTNH->BFNH", weights, value)

    attention_output = dense_layer('output_transform', attention_output,
                     subscripts='abcd,cde->abe',
                     use_bias=False)
  return attention_output

def feed_forward_network(x):
  with variable_scope("feed_forward_network"):
    output = dense_layer('filter_layer', x, subscripts='abc,cd->abd', activation=tf.nn.relu)
    output = dense_layer('output_layer', output, subscripts='abc,cd->abd')
  return output

# Transformer

def encoder_stack(encoder_inputs, num_layers=6):
  with variable_scope('encoder_stack'):
    for n in range(num_layers):
      with variable_scope("layer_%d" % n):
        with variable_scope("self_attention"):
          encoder_inputs = pre_post_processing_wrapper(
            self_attention_layer,
            encoder_inputs,
            None)
        with variable_scope("ffn"):
          encoder_inputs = pre_post_processing_wrapper(
            feed_forward_network,
            encoder_inputs)

    return layer_norm(encoder_inputs, epsilon=1e-6)

def decoder_stack(decoder_inputs,
          encoder_outputs,
          decoder_self_attention_bias,
          num_layers=6):
  with variable_scope('decoder_stack'):
    for n in range(num_layers):
      with variable_scope("layer_%d" % n):
        with variable_scope("self_attention"):
          decoder_inputs = pre_post_processing_wrapper(
            self_attention_layer,
            decoder_inputs,
            decoder_self_attention_bias)
        with variable_scope("encdec_attention"):
          decoder_inputs = pre_post_processing_wrapper(
            attention_layer,
            decoder_inputs,
            encoder_outputs,
            None)
        with variable_scope("ffn"):
          decoder_inputs = pre_post_processing_wrapper(
            feed_forward_network,
            decoder_inputs)

    return layer_norm(decoder_inputs, epsilon=1e-6)

def encode(inputs, hidden_size=512):
  with variable_scope('encode'):
    embedded_inputs = embedding_softmax_layer(inputs)
    length = tf.shape(embedded_inputs)[1]
    pos_encoding = get_position_encoding(length, hidden_size)
    
    encoder_inputs = embedded_inputs + pos_encoding

    return encoder_stack(encoder_inputs)

def decode(targets, encoder_outputs, hidden_size=512):
  with variable_scope("encode"):
    enbedded_inputs = embedding_softmax_layer(targets[:, :-1])

  with variable_scope("decode"):
    length = tf.shape(enbedded_inputs)[1]
    pos_encoding = get_position_encoding(length, hidden_size)
    decoder_inputs = enbedded_inputs + pos_encoding

    decoder_self_attention_bias = get_decoder_self_attention_bias(length)
    outputs = decoder_stack(
      decoder_inputs,
      encoder_outputs,
      decoder_self_attention_bias)

  with variable_scope("encode"):
    logits = embedding_softmax_layer(outputs, mode="linear")

  return logits

def body(inputs, targets):
  encoder_outputs = encode(inputs)
  logits = decode(targets, encoder_outputs)
  return logits

class Transformer(tf.keras.layers.Layer):
  def call(self, inputs, targets, training):
    encoder_outputs = encode(inputs)
    logits = decode(targets, encoder_outputs)
    return logits

def load_model(file):
  arr = np.load(file)
  set_variables(arr)
  @tf.function(input_signature=(
    tf.TensorSpec(shape=[1, 20], dtype=tf.int64),
    tf.TensorSpec(shape=[1, 20], dtype=tf.int64),)
  )
  def f(inputs, targets):
    return body(inputs, targets)
  return f
