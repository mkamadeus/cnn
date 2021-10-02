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


def test_backprop_dense_1():
    layer = Dense(size=2, input_size=2, weights=np.array([[0, 0], [1, 2], [3, -4]]), activation="relu")
    layer.run(inputs=np.array([118, 102]))
    result = layer.compute_delta(np.array([24.0, 10.0]))
    expected = np.array([[2832.0, 0.0], [2448.0, 0.0]])

    assert np.testing.assert_array_almost_equal(result, expected) is None


def test_backprop_dense_2():
    layer = Dense(
        size=10,
        input_size=2,
        weights=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.04, 0.05, 0.01],
                [0.02, 0.03, 0.02, 0.02, 0.01, 0.02, 0.07, 0.08, 0.05, 0.01],
            ]
        ),
        activation="softmax",
    )
    layer.run(inputs=np.array([424, 0]))
    first_delta = softmax_derivative(
        output=np.array(
            [
                9.855924e-01,
                1.420001e-02,
                2.045880e-04,
                2.947620e-06,
                4.246811e-08,
                6.118632e-10,
                8.815475e-12,
                6.118632e-10,
                4.246811e-08,
                1.829905e-15,
            ]
        ),
        target_class=9,
    )
    result = layer.compute_delta(first_delta)
    ic(first_delta)
    expected = np.array(
        [
            [4.18e02, 0.00e00],
            [6.02e00, 0.00e00],
            [8.67e-02, 0.00e00],
            [1.25e-03, 0.00e00],
            [1.80e-05, 0.00e00],
            [2.59e-07, 0.00e00],
            [3.74e-09, 0.00e00],
            [2.59e-07, 0.00e00],
            [1.80e-05, 0.00e00],
            [-4.24e02, 0.00e00],
        ]
    ).T
    assert np.testing.assert_allclose(result, expected, rtol=1e-02) is None
