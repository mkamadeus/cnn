import numpy as np


class BaseLayer:
    """
    Defines base class for a layer.
    """

    def __init__(self):
        self.type = "base"
        return

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def get_type(self):
        return self.type

    def get_shape(self, input_shape):
        return input_shape

    def get_weight_count(self):
        return 0

    def get_shape_and_weight_count(self, input_shape):
        return self.get_shape(input_shape), self.get_weight_count()
