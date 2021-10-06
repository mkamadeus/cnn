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

    def run(self, inputs: np.ndarray):
        # store result
        self.result = inputs
        return self.result

    def predict(self):
        # get prediction
        # output = np.zeros(self.size)
        # ic(np.argmax(self.result))
        # output[np.argmax(self.result)] = 1.0

        return self.result

    def compute_delta(self, delta: np.ndarray) -> np.ndarray:
        if self.error_mode == "logloss":
            return self.result - delta
        elif self.error_mode == "sse":
            return self.result - delta
