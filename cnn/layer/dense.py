import numpy as np
from cnn.activations import sigmoid, relu, softmax
from cnn.layer.base import BaseLayer

ACTIVATION_MODES = ["relu", "sigmoid", "softmax"]


class DenseLayer(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, weights, activation="sigmoid"):
        if activation not in ACTIVATION_MODES:
            raise Exception("invalid pooling mode")

        self.size: int = size
        self.weights = weights
        self.activation = activation

    def run(self, inputs: np.array) -> np.ndarray:
        if self.activation == "sigmoid":
            activation_func = sigmoid
        elif self.activation == "relu":
            activation_func = relu
        elif self.activation == "softmax":
            activation_func = softmax(inputs)
        else:
            raise Exception("invalid activation mode")

        # add bias to input
        biased_input: np.ndarray = np.insert(inputs, 0, 1)

        result = np.matmul(self.weights, biased_input.T).flatten()
        result = activation_func(result)

        return result
