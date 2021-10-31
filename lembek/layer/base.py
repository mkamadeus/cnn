import numpy as np


class BaseLayer:
    """
    Defines base class for a layer.
    """

    def __init__(self):
        self.type = "base"
        return

    def compute_delta(self, delta: np.ndarray) -> np.ndarray:
        return delta

    def update_weights(self, learning_rate: float, momentum: float):
        pass

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def get_type(self):
        return self.type

    def get_X(self):
        return self.X

    def get_W(self):
        return self.W

    def get_shape(self, input_shape=None):
        if input_shape is None:
            n_channel = len(self.inputs[0])
            length = len(self.inputs[0][0])
            width = len(self.inputs[0][0][0])
            input_shape = (n_channel, length, width)
        return input_shape

    def get_weight_count(self):
        return 0

    def get_shape_and_weight_count(self, input_shape):
        return self.get_shape(input_shape), self.get_weight_count()
