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


def test_backprop_dense():
    layer = Dense(size=2, input_size=2, weights=np.array([[0, 0], [1, 2], [3, -4]]), activation="relu")
    layer.run(inputs=np.array([118, 102]))
    result = layer.compute_delta(np.array([24.0, 10.0]))
    expected = np.array([[2832.0, 0.0], [2448.0, 0.0]])

    assert np.testing.assert_array_almost_equal(result, expected) is None
