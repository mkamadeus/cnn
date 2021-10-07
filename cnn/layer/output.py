import numpy as np
from cnn.layer import BaseLayer

# from icecream import ic

ACTIVATION_MODES = ["relu", "sigmoid", "linear", "softmax"]
ERROR_MODES = ["sse", "logloss"]


class Output(BaseLayer):
    """
    Defines an output layer.
    """

    def __init__(self, size: int, error_mode: str = "logloss"):
        if error_mode not in ERROR_MODES:
            raise ValueError("invalid error mode")

        self.size = size
        self.error_mode = error_mode
        self.type = "output"

    def run(self, inputs: np.ndarray):
        # store result
        self.result = inputs
        return self.result

    def predict(self, activation: str, sigmoid_threshold: float = 0.5):
        # get prediction
        if activation not in ACTIVATION_MODES:
            raise ValueError("invalid activation mode")
        if activation == "softmax":
            return np.array([np.argmax(self.result)])
        if activation == "sigmoid":
            return np.argwhere(self.result >= sigmoid_threshold).flatten()

        return self.result

    def compute_delta(self, delta: np.ndarray) -> np.ndarray:
        if self.error_mode == "logloss":
            return self.result - delta
        elif self.error_mode == "sse":
            return self.result - delta
