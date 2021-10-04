import numpy as np
from cnn.layer import Output


def test_linear_output():
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    model = Output(size=5, activation="linear")
    result = model.run(inputs)

    assert np.testing.assert_array_equal(result, inputs) is None


def test_relu_output():
    inputs = np.array([0.0, 0.0, 0.0, 3.0, 5.0])

    model = Output(size=5, activation="relu")
    result = model.run(inputs)

    assert np.testing.assert_array_equal(result, inputs) is None


def test_softmax_output():
    inputs = np.array(
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
    )

    model = Output(size=10, activation="softmax")
    result = model.run(inputs)
    expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sigmoid_output():
    inputs = np.array([0.32, 0.6, 0.4, 0.7, 0.99, 0.6, 0.3, 0.25, 0.15, 0.45])

    model = Output(size=10, activation="sigmoid")
    result = model.run(inputs)
    expected = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    assert np.testing.assert_array_equal(result, expected) is None
