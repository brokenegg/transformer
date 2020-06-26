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
    x_shift = x - np.expand_dims(np.mean(x, axis=-1), axis=-1)
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

def embedding_softmax_layer(inputs, hidden_size=512):
    with variable_scope('embedding_shared_weights'):
        with variable_scope('embedding_and_softmax'):
            weights = get_variable('weights')
            embedded_inputs = weights[inputs]
            embedded_inputs *= hidden_size ** 0.5
    return embedded_inputs

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

def pre_post_processing_wrapper(x, layer):
    with variable_scope("pre_post_processing_wrapper"):
        y = layer_norm(x)
        y = layer(y)
        return x + y
        
def feed_forward_network(x):
    with variable_scope("feed_forward_network"):
        output = dense_layer('filter_layer', x, subscripts='abc,cd->abd', activation=relu)
        output = dense_layer('output_layer', output, subscripts='abc,cd->abd')
    return output