import numpy as np
from cnn.layer import Dense
from cnn.activations import softmax_derivative
from icecream import ic


def test_dense():
    layer = Dense(
        size=3,
        input_size=3,
        weights=np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]),
    )
    result = layer.run(inputs=np.array([1, 2, 3]))
    expected = np.array([0.890903, 0.802184, 0.668187])

    assert np.testing.assert_array_almost_equal(result, expected, decimal=6) is None


def test_compute_delta_dense_1():
    layer = Dense(size=2, input_size=2, weights=np.array([[0, 0], [1, 2], [3, -4]]), activation="relu")
    layer.run(inputs=np.array([118, 102]))
    result = layer.compute_delta(np.array([[24.0], [10.0]]))
    expected = np.array([[24.0], [72.0]])

    assert np.testing.assert_array_almost_equal(result, expected) is None


def test_compute_delta_dense_2():
    layer = Dense(
        size=10,
        input_size=2,
        weights=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.01],
                [0.02, 0.03, 0.03, 0.02, 0.01, 0.02, 0.07, 0.08, 0.05, 0.01],
            ]
        ),
        activation="softmax",
    )
    output = layer.run(inputs=np.array([424, 0]))
    first_delta = softmax_derivative(
        output=output,
        target_class=9,
    )
    result = layer.compute_delta(first_delta.reshape(len(first_delta), 1))
    ic(first_delta.reshape(len(first_delta), 1))
    expected = np.array(
        [
            [0.07985382],
            [0.01014405],
        ]
    )
    assert np.testing.assert_allclose(result, expected, rtol=1e-06) is None
