import unittest
import torch
import numpy as np
from . import transformer_numpy
from . import transformer_torch

class TestTransformer(unittest.TestCase):

    def assertArrayAlmostEqual(self, a, b, max_delta=1e-5):
        if isinstance(a, torch.Tensor):
            a = a.detach().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().numpy()
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(list(a.shape), list(b.shape))
        delta = np.sqrt(np.sum((a - b) ** 2))
        self.assertLessEqual(delta, max_delta)

    def test_einsum_1(self):
        a = torch.rand(3,4,5,6)
        b = torch.rand(3,7,5,6)
        c = transformer_torch.einsum("BTNH,BFNH->BNFT", a, b)
        d =  np.einsum("BTNH,BFNH->BNFT", a, b)
        self.assertArrayAlmostEqual(c, d)

    def test_einsum_2(self):
        a = torch.rand(3,4,5,6)
        b = torch.rand(5,6,7)
        c = transformer_torch.einsum('abcd,cde->abe', a, b)
        d =  np.einsum('abcd,cde->abe', a, b)
        self.assertArrayAlmostEqual(c, d)

    def test_get_position_encoding(self):
        a = transformer_torch.get_position_encoding(10, 512)
        b = transformer_numpy.get_position_encoding(10, 512)
        self.assertArrayAlmostEqual(a, b)

    def test_encode(self):
        if False:
            return
        arr = np.load('examples/brokenegg.npz')
        inputs = np.array([[6086, 33, 631, 769, 489, 3, 2]], dtype=np.long)
        targets = np.array([[1, 123, 37, 166, 120, 16147, 712, 123, 37, 166, 18, 7, 489, 3, 2]], dtype=np.long)

        transformer_numpy.set_variables(arr)
        outputs_numpy = transformer_numpy.body(inputs, targets)

        transformer_torch.set_variables(arr)
        outputs_torch = transformer_torch.body(torch.tensor(inputs), torch.tensor(targets)[:, :-1])
        
        self.assertArrayAlmostEqual(outputs_numpy, outputs_torch, max_delta=1e-2)
