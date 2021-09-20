import numpy as np
from cnn.layer import Dense


def test_dense():
    layer = Dense(
        size=3,
        input_size=3,
        weights=np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
    )
    result = layer.run(inputs=np.array([1, 2, 3]))
    expected = np.array([0.890903, 0.802184, 0.668187])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None
