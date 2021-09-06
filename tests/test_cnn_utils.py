import unittest
import numpy as np
from cnn.utils import generate_strides, pad_array
from numpy.testing import assert_array_equal


class TestCNNUtils(unittest.TestCase):
    def test_pad_array_1(self):
        # makes 1,2,3,4 as a 2x2 matrix
        mat = (np.arange(4) + 1).reshape((2, 2))

        result = pad_array(mat, 1, 0)
        expected = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 0, 0, 0],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))

    def test_pad_array_2(self):
        # makes 1,2,3,4 as a 2x2 matrix
        mat = (np.arange(4) + 1).reshape((2, 2))

        result = pad_array(mat, 2, 0)
        expected = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 2, 0, 0],
                [0, 0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))

    def test_generate_strides_1(self):
        # makes 1..9 as a 3x3 matrix
        mat = (np.arange(9) + 1).reshape((3, 3))

        result = generate_strides(mat, (2, 2))
        expected = np.array(
            [
                [
                    [1, 2],
                    [4, 5],
                ],
                [
                    [2, 3],
                    [5, 6],
                ],
                [
                    [4, 5],
                    [7, 8],
                ],
                [
                    [5, 6],
                    [8, 9],
                ],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))
