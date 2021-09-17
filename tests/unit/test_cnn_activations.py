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


def test_sigmoid():
    result = sigmoid(0)
    expected = 0.5

    assert result == expected


def test_sigmoid_derivative():
    result = sigmoid_derivative(0.5)
    expected = 0.25

    assert result == expected


def test_linear():
    for i in range(-5, 5):
        result = linear(i)
        expected = i

        assert result == expected


def test_linear_derivative():
    for i in range(-5, 5):
        result = linear_derivative(i)
        expected = 1

        assert result == expected


def test_relu_1():
    result = relu(-999)
    expected = 0

    assert result == expected


def test_relu_2():
    result = relu(999)
    expected = 999

    assert result == expected


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
