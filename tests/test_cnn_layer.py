from cnn.layer import ConvolutionalLayer
import unittest
import numpy as np
from numpy.testing import assert_array_equal


class TestCNNLayer(unittest.TestCase):
    # TODO: multiple channels, multiple kernels
    def test_convolution(self):
        layer = ConvolutionalLayer(
            kernel=np.array(
                [
                    [1, 1],
                    [0, 1],
                ]
            ),
            stride=1,
            padding=0,
        )
        feature_map = layer.convolution(1, 0)
        expected = np.array(
            [
                [9, 32],
                [14, 26],
            ]
        )
        self.assertIsNone(assert_array_equal(feature_map, expected))
