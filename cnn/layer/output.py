import numpy as np
from cnn.layer import BaseLayer
from icecream import ic

ACTIVATION_MODES = ["relu", "sigmoid", "linear", "softmax"]


class Output(BaseLayer):
    """
    Defines an output layer.
    """

    def __init__(self, size: int, activation: str, sigmoid_threshold: float = 0.5):
        self.size = size
        self.activation = activation
        self.sigmoid_threshold = sigmoid_threshold

    def run(self, inputs: np.ndarray):
        if self.activation == "softmax":
            output = np.zeros(self.size)
            ic(np.argmax(inputs))
            output[np.argmax(inputs)] = 1.0
            return output

        if self.activation == "sigmoid":
            func = np.vectorize(lambda x: 1.0 if x >= self.sigmoid_threshold else 0.0)
            return func(inputs)

        return inputs
