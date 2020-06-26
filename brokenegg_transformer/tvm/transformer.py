# Copyright Katsuya Iida.

import numpy as np
import math

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
    trainable_variables = vars

def list_variables():
    return [
        (k, v.shape)
        for k, v in trainable_variables.items()
    ]

# Misc

def relu(x):
    return (x >= 0).astype(np.float32) * x
    
def softmax(x):
    x_shift = x - np.expand_dims(np.max(x, axis=-1), axis=-1)
    exp_x = np.exp(x_shift)
    s = np.sum(exp_x, axis=-1)
    return exp_x / np.expand_dims(s, axis=-1)

def layer_norm(inputs, epsilon=0.001):
    with variable_scope('layer_normalization'):
        mean = np.expand_dims(np.mean(inputs, axis=-1), axis=-1) 
        var = np.expand_dims(np.var(inputs, axis=-1), axis=-1)
        norm_inputs = (inputs - mean) / np.sqrt(var + epsilon)
        gamma = get_variable('gamma')
        beta = get_variable('beta')
        return gamma * norm_inputs + beta

def dense_layer(name, inputs, subscripts='abc,cde->abde', use_bias=True, activation=None):
    with variable_scope(name):
        kernel = get_variable('kernel')
        y = np.einsum(subscripts, inputs, kernel)
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
                embedded_inputs = shared_weights[inputs]
                embedded_inputs *= hidden_size ** 0.5
                return embedded_inputs
            elif mode == "linear":
                vocab_size = shared_weights.shape[0]
                batch_size = inputs.shape[0]
                length = inputs.shape[1]
                x = np.reshape(inputs, [-1, hidden_size])
                logits = np.matmul(x, shared_weights.T, )

                return np.reshape(logits, [batch_size, length, vocab_size])

def get_decoder_self_attention_bias(length):
    neg_inf = -1e9
    valid_locs = np.tri(length)
    valid_locs = np.reshape(valid_locs, [1, 1, length, length])
    decoder_bias = neg_inf * (1.0 - valid_locs)
    return decoder_bias

def get_position_encoding(
      length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length).astype(np.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (np.array(num_timescales, np.float32) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
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

        logits = np.einsum("BTNH,BFNH->BNFT", key, query)
        if bias is not None:
            logits += bias
        weights = softmax(logits)
        attention_output = np.einsum("BNFT,BTNH->BFNH", weights, value)

        attention_output = dense_layer('output_transform', attention_output,
                                       subscripts='abcd,cde->abe',
                                       use_bias=False)
    return attention_output

def feed_forward_network(x):
    with variable_scope("feed_forward_network"):
        output = dense_layer('filter_layer', x, subscripts='abc,cd->abd', activation=relu)
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
        length = embedded_inputs.shape[1]
        pos_encoding = get_position_encoding(length, hidden_size)
        
        encoder_inputs = embedded_inputs + pos_encoding

        return encoder_stack(encoder_inputs)

def decode(targets, encoder_outputs, hidden_size=512):
    with variable_scope("encode"):
        enbedded_inputs = embedding_softmax_layer(targets[:, :-1])

    with variable_scope("decode"):
        length = enbedded_inputs.shape[1]
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
