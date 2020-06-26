import numpy as np

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
    
def relu(x):
    return (x >= 0).astype(np.float32) * x
    
def softmax(x):
    exp_x = np.exp(x)
    s = np.sum(exp_x, axis=-1)
    return exp_x / np.expand_dims(s, axis=len(x.shape)-1)

def layer_norm(inputs, epsilon=0.001):
    with variable_scope('layer_normalization'):
        mean = np.expand_dims(np.mean(inputs, axis=-1), axis=-1) 
        var = np.expand_dims(np.var(inputs, axis=-1), axis=-1)
        norm_inputs = (inputs - mean) / np.sqrt(var + epsilon)
        gamma = get_variable('gamma')
        beta = get_variable('beta')
        return gamma * norm_inputs + beta