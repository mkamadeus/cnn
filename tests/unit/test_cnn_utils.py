import numpy as np
from cnn.utils import (
    generate_strides,
    pad_array,
    generate_random_uniform_matrixes,
    add_all_feature_maps,
)
from numpy.testing import assert_array_equal


def test_pad_array_1():
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
    assert assert_array_equal(result, expected) is None


def test_pad_array_2():
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
    assert assert_array_equal(result, expected) is None


def test_generate_strides_1():
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
    assert_array_equal(result, expected) is None


def test_generate_strides_2():
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
    assert assert_array_equal(result, expected) is None


def test_generate_strides_padded():
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

    assert assert_array_equal(padded, expected_padded) is None

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
    assert assert_array_equal(result, expected) is None


def test_generate_random_matrixes_1():
    # makes 2x2 random kernel matrix
    # for a conv layer with 3 filters and 3 channels
    n_filter = 3
    n_channel = 3
    size = (2, 2)

    result = generate_random_uniform_matrixes(n_filter, n_channel, size)

    assert result.shape == (3, 3, 2, 2)
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_add_all_feature_maps():
    feature_map_arr = np.array(
        [[[0, 1], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
    )

    expected = np.array(
        [
            [14, 17],
            [21, 24],
        ]
    )

    result = add_all_feature_maps(feature_map_arr)

    assert assert_array_equal(result, expected) is None
