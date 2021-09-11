import unittest
import numpy as np
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes
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
                    [
                        [1, 2],
                        [4, 5],
                    ],
                    [
                        [2, 3],
                        [5, 6],
                    ],
                ],
                [
                    [
                        [4, 5],
                        [7, 8],
                    ],
                    [
                        [5, 6],
                        [8, 9],
                    ],
                ],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))

    def test_generate_strides_2(self):
        # makes 1..16 as a 4x4 matrix
        mat = (np.arange(16) + 1).reshape((4, 4))

        result = generate_strides(mat, (2, 2), 2)
        expected = np.array(
            [
                [
                    [
                        [1, 2],
                        [5, 6],
                    ],
                    [
                        [3, 4],
                        [7, 8],
                    ],
                ],
                [
                    [
                        [9, 10],
                        [13, 14],
                    ],
                    [
                        [11, 12],
                        [15, 16],
                    ],
                ],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))

    def test_generate_strides_padded(self):
        # makes 1..16 as a 4x4 matrix
        mat = (np.arange(4) + 11).reshape((2, 2))
        padded = pad_array(mat, 1, 0)
        expected_padded = np.array(
            [
                [0, 0, 0, 0],
                [0, 11, 12, 0],
                [0, 13, 14, 0],
                [0, 0, 0, 0],
            ]
        )
        self.assertIsNone(assert_array_equal(padded, expected_padded))

        result = generate_strides(expected_padded, (2, 2), 2)
        expected = np.array(
            [
                [
                    [
                        [0, 0],
                        [0, 11],
                    ],
                    [
                        [0, 0],
                        [12, 0],
                    ],
                ],
                [
                    [
                        [0, 13],
                        [0, 0],
                    ],
                    [
                        [14, 0],
                        [0, 0],
                    ],
                ],
            ]
        )
        self.assertIsNone(assert_array_equal(result, expected))

    def test_generate_random_matrixes_1(self):
        # makes 3 random uniform 2x2 matrixes
        n_matrix = 3
        size = (2, 2)

        result = generate_random_uniform_matrixes(n_matrix, size)

        self.assertEqual(result.shape, (3, 2, 2))
        self.assertTrue(result.min() >= -1.0)
        self.assertTrue(result.max() <= 1.0)
