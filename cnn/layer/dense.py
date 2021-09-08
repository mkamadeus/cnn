import numpy as np
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

        # TODO: activation functions
        self.activation = lambda x: x

    def run(self, inputs: np.array) -> np.ndarray:
        # add bias to input
        biased_input: np.ndarray = np.insert(inputs, 0, 1)

        result = np.matmul(self.weights, biased_input.T).flatten()
        result = self.activation(result)

        return result
