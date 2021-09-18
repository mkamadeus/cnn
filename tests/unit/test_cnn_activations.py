from cnn.activations import (
    sigmoid,
    sigmoid_derivative,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
    softmax,
)
import numpy as np


def test_sigmoid_1():
    result = sigmoid(np.array([0]))
    expected = np.array([0.5])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sigmoid_2():
    result = sigmoid(np.array([1, 2, 3]))
    expected = np.array([0.73105858, 0.88079708, 0.95257413])

    assert np.testing.assert_array_almost_equal(result, expected) is None


def test_sigmoid_derivative_1():
    result = sigmoid_derivative(np.array([0.5]))
    expected = np.array([0.25])

    assert np.testing.assert_array_equal(result, expected) is None


def test_sigmoid_derivative_2():
    result = sigmoid_derivative(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    expected = np.array([0.09, 0.16, 0.21, 0.24, 0.25])

    assert np.testing.assert_array_almost_equal(result, expected) is None


def test_linear():
    data = range(-5, 5)
    result = linear(data)

    assert np.testing.assert_array_equal(result, data) is None


def test_linear_derivative():
    data = range(-5, 5)
    expected = [1 for _ in range(-5, 5)]
    result = linear_derivative(data)

    assert np.testing.assert_array_equal(result, expected) is None


def test_relu_1():
    result = relu(np.array([-999]))
    expected = np.array([0])

    assert np.testing.assert_array_equal(result, expected) is None


def test_relu_2():
    result = relu(np.array([999]))
    expected = np.array([999])

    assert np.testing.assert_array_equal(result, expected) is None


def test_relu_derivative_1():
    result = relu_derivative(-999)
    expected = 0

    assert result == expected


def test_relu_derivative_2():
    result = relu_derivative(999)
    expected = 1

    assert result == expected


def test_softmax():
    layer = np.array([1, 2, 3])

    softmax_fun = softmax(layer)

    assert np.sum(softmax_fun(layer)) == 1.0
