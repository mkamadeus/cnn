from cnn.layer import ConvolutionalLayer
import unittest
import numpy as np
from numpy.testing import assert_array_equal


class TestCNNLayer(unittest.TestCase):
    # TODO: multiple channels, multiple kernels
    def test_convolution(self):
        layer = ConvolutionalLayer(
            np.array(
                [
                    [1, 7, 2],
                    [11, 1, 23],
                    [2, 2, 2],
                ]
            ),
            np.array(
                [
                    [1, 1],
                    [0, 1],
                ]
            ),
        )
        feature_map = layer.convolution(1, 0)
        expected = np.array(
            [
                [9, 32],
                [14, 26],
            ]
        )
        self.assertIsNone(assert_array_equal(feature_map, expected))
