import numpy as np
from cnn.activations import linear, relu, sigmoid
from cnn.layer.base import BaseLayer

DETECTOR_MODES = ["relu", "sigmoid", "linear"]


class Detector(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self, activation: str):
        if activation not in DETECTOR_MODES:
            raise ValueError("invalid activation mode")

        self.activation = activation
        self.type = "detector     "

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        if self.activation == "relu":
            act_f = relu
        elif self.activation == "sigmoid":
            act_f = sigmoid
        elif self.activation == "linear":
            act_f = linear

        act_func = np.vectorize(act_f, otypes=[float])
        return act_func(inputs)

    def compute_delta(self, delta: np.ndarray):
        return delta

    def update_weights(self, learning_rate: float):
        pass
