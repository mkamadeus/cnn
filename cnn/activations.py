import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    return x * (x.fill(1) - x)


def linear(x: np.ndarray):
    return x


def linear_derivative(x: np.ndarray):
    return x.fill(1)


def relu(x: np.ndarray):
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray):
    return (x > 0).astype(float)


def softmax(layer: np.ndarray):
    return lambda x: np.exp(x) / np.sum(np.exp(layer))


def softmax_derivative():
    pass
