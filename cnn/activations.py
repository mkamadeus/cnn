import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def linear(x):
    return x


# TO-DO: the x is array-like input
def linear_derivative(x):
    return 1


def relu(x):
    return np.maximum(0, x)


# TO-DO: the x is array-like input
def relu_derivative(x):
    return 1 if x > 0 else 0


def softmax(layer):
    return lambda x: np.exp(x) / np.sum(np.exp(layer))


def softmax_derivative():
    pass
