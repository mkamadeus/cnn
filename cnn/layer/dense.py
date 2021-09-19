import numpy as np
from cnn.activations import sigmoid, relu, softmax
from cnn.layer.base import BaseLayer
from icecream import ic

ACTIVATION_MODES = ["relu", "sigmoid", "softmax"]


class Dense(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, weights=None, activation="sigmoid"):
        if activation not in ACTIVATION_MODES:
            raise Exception("invalid pooling mode")

        self.type = "dense"
        self.size: int = size
        self.weights = weights
        if self.weights is None:
            self.weights = np.random.random((self.size, np.random.randint(1, 10)))
        self.activation = activation

    def run(self, inputs: np.array) -> np.ndarray:
        # add bias to input
        biased_input: np.ndarray = np.insert(inputs, 0, 1)
        ic(inputs)

        result = np.matmul(self.weights.T, biased_input).flatten()
        ic(result)

        if self.activation == "sigmoid":
            activation_func = sigmoid
        elif self.activation == "relu":
            activation_func = relu
        elif self.activation == "softmax":
            activation_func = softmax(result)
        else:
            raise Exception("invalid activation mode")

        result = activation_func(result)
        ic(result)

        return result

    def get_shape(self, input_shape=None):
        return (1, 1, self.size)

    def get_weight_count(self):
        return len(self.weights)
