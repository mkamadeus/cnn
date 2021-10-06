import numpy as np
from cnn.layer import Output


def test_logloss_output():
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    model = Output(size=5)
    result_forward_prop = model.run(inputs)

    assert np.testing.assert_array_equal(result_forward_prop, inputs) is None


def test_sse_output():
    inputs = np.array([0.0, 0.0, 0.0, 3.0, 5.0])

    model = Output(size=5, error_mode="sse")
    result = model.run(inputs)

    assert np.testing.assert_array_equal(result, inputs) is None


def test_logloss_compute_delta():
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    model = Output(size=5)
    model.run(inputs)

    result = model.compute_delta(target)
    expected = np.array([0.0, 2.0, 3.0, 4.0, 5.0])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sse_compute_delta():
    inputs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    model = Output(size=5, error_mode="sse")
    model.run(inputs)

    result = model.compute_delta(target)
    expected = np.array([0.0, 2.0, 3.0, 4.0, 5.0])

    assert np.testing.assert_array_equal(result, expected) is None


def test_softmax_predict():
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

    model = Output(size=10)
    model.run(inputs)

    result = model.predict(activation="softmax")
    expected = np.array([0])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sigmoid_predict():
    inputs = np.array([0.32, 0.6, 0.4, 0.7, 0.99, 0.6, 0.3, 0.25, 0.15, 0.45])

    model = Output(size=10, error_mode="sse")
    model.run(inputs)

    result = model.predict(activation="sigmoid")
    expected = np.array([1, 3, 4, 5])

    assert np.testing.assert_array_equal(result, expected) is None


def test_linear_predict():
    inputs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = Output(size=5, error_mode="sse")
    model.run(inputs)

    result = model.predict(activation="linear")

    assert np.testing.assert_array_equal(result, inputs) is None


def test_relu_predict():
    inputs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    model = Output(size=5, error_mode="sse")
    model.run(inputs)

    result = model.predict(activation="relu")

    assert np.testing.assert_array_equal(result, inputs) is None
