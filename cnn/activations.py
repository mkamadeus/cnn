import numpy as np
# from numpy import float128


def sigmoid(x: np.ndarray):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray):
    return x * (np.full((x.shape), 1) - x)


def linear(x: np.ndarray):
    return x


def linear_derivative(x: np.ndarray):
    return np.full((x.shape), 1)


def relu(x: np.ndarray):
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray):
    return (x > 0).astype(float)


def softmax(layer: np.ndarray):
    f = np.exp(layer - np.max(layer))
    return lambda x: np.exp(x) / np.sum(np.exp(f))

def softmax_derivative(output: np.ndarray, target_class: int):
    """
    This is actually the derivative of negative log likelihood loss and softmax activation function.
    """
    target_arr = np.zeros(output.shape)
    target_arr[target_class] = 1.0
    return output - target_arr
