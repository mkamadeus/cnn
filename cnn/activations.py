import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def softmax(x, layer):
    return np.exp(x) / np.sum(np.exp(layer))


def softmax_derivative():
    pass
