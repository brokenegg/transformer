import unittest
import tensorflow as tf
import numpy as np
from .transformer_numpy import *

class TestTransformer(unittest.TestCase):

    def assertArrayAlmostEqual(self, a, b, max_delta=1e-5):
        delta = np.sqrt(np.sum((a - b) ** 2))
        print(delta)
        self.assertLessEqual(delta, max_delta)

    def test_relu(self):
        x = np.random.normal(scale=10.0, size=(5, 3, 7)).astype(np.float32)
        y = relu(x)
        #print(y)
        z = tf.nn.relu(x)
        self.assertArrayAlmostEqual(y, z)

    def test_softmax(self):
        x = np.random.normal(scale=10.0, size=(5, 3, 7)).astype(np.float32)
        y = softmax(x)
        print(y)
        z = tf.nn.softmax(x)
        self.assertArrayAlmostEqual(y, z)

    def test_layer_norm(self):
        gamma = np.random.normal(scale=10.0, size=(7,)).astype(np.float32)
        beta = np.random.normal(scale=10.0, size=(7,)).astype(np.float32)
        x = np.random.normal(scale=10.0, size=(5, 3, 7)).astype(np.float32)

        with variable_scope('test'):
            with variable_scope('layer_normalization'):
                set_variable('gamma', gamma)
                set_variable('beta', beta)
            y = layer_norm(x, epsilon=1e-5)

        tf_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, dtype="float32")
        tf_layer_norm(x)
        tf_layer_norm.gamma.assign(gamma)
        tf_layer_norm.beta.assign(beta)
        z = tf_layer_norm(x)

        self.assertArrayAlmostEqual(y, z, max_delta=1e-4)
