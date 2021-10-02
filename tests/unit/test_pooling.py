from cnn.layer.pooling import MaxPooling, AveragePooling
import pytest
import numpy as np


@pytest.mark.parametrize(
    "pool_input,pool_output,size,stride",
    [
        (np.arange(1, 17).reshape((1, 4, 4)), np.array([[[6, 7, 8], [10, 11, 12], [14, 15, 16]]]), (2, 2), 1),
        (
            np.array([[[1, 1, 1, 1], [1, 99, 99, 1], [1, 99, 99, 1], [1, 1, 1, 1]]]),
            np.array([[[99, 99], [99, 99]]]),
            (3, 3),
            1,
        ),
    ],
)
def test_max_pooling_forward(pool_input, pool_output, size, stride):
    layer = MaxPooling(size=size, stride=stride)
    result = layer.run(pool_input)
    assert np.testing.assert_array_almost_equal(result, pool_output) is None
    pass


@pytest.mark.parametrize(
    "pool_input,pool_output,size,stride",
    [
        (
            np.arange(1, 17).reshape((1, 4, 4)),
            np.array([[[3.5, 4.5, 5.5], [7.5, 8.5, 9.5], [11.5, 12.5, 13.5]]]),
            (2, 2),
            1,
        ),
        (
            np.array([[[1, 1, 1, 1], [1, 99, 99, 1], [1, 99, 99, 1], [1, 1, 1, 1]]]),
            np.array([[[44.555556, 44.555556], [44.555556, 44.555556]]]),
            (3, 3),
            1,
        ),
    ],
)
def test_average_pooling_forward(pool_input, pool_output, size, stride):
    layer = AveragePooling(size=size, stride=stride)
    result = layer.run(pool_input)
    assert np.testing.assert_array_almost_equal(result, pool_output) is None
    pass


@pytest.mark.parametrize(
    "pool_input,pool_output,delta_input,delta_output,size,stride",
    [
        (
            np.arange(1, 17).reshape((1, 4, 4)),
            np.array([[[6, 7, 8], [10, 11, 12], [14, 15, 16]]]),
            np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]),
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 0.1, 0.2, 0.3],
                        [0, 0.4, 0.5, 0.6],
                        [0, 0.7, 0.8, 0.9],
                    ]
                ]
            ),
            (2, 2),
            1,
        ),
        (
            np.array([[[1, 1, 1, 1], [1, 99, 99, 1], [1, 99, 99, 1], [1, 1, 1, 1]]]),
            np.array([[[99, 99], [99, 99]]]),
            np.array([[[0.1, 0.2], [0.3, 0.4]]]),
            np.array(
                [
                    [
                        [0, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ]
            ),
            (3, 3),
            1,
        ),
    ],
)
def test_pooling_backward(pool_input, pool_output, delta_input, delta_output, size, stride):
    layer = MaxPooling(size=size, stride=stride)
    layer.input = pool_input
    layer.output = pool_output

    delta = layer.compute_delta(delta_input)

    assert np.testing.assert_array_almost_equal(delta, delta_output, decimal=4) is None
    pass
