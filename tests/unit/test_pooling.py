from cnn.layer import Pooling
import pytest
import numpy as np


@pytest.mark.parametrize(
    "pool_input,pool_output,size,stride,mode",
    [
        (np.arange(1, 17).reshape((1, 4, 4)), np.array([[[6, 7, 8], [10, 11, 12], [14, 15, 16]]]), (2, 2), 1, "max"),
        (
            np.array([[[1, 1, 1, 1], [1, 99, 99, 1], [1, 99, 99, 1], [1, 1, 1, 1]]]),
            np.array([[[99, 99], [99, 99]]]),
            (3, 3),
            1,
            "max",
        ),
    ],
)
def test_pooling_forward(pool_input, pool_output, size, stride, mode):
    print(pool_input)
    layer = Pooling(size=size, stride=stride, mode=mode)
    result = layer.run(pool_input)
    assert np.testing.assert_array_equal(result, pool_output) is None
    pass
