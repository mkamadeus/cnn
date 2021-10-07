import numpy as np
from cnn.layer.base import BaseLayer


class Flatten(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self):
        self.type = "flatten"
        self.shape_before = None

    def run(self, inputs: np.array):
        self.shape_before = inputs.shape
        flatten_output = inputs.flatten()
        return flatten_output

    def get_shape(self, input_shape=None):
        input_shape = super().get_shape(input_shape)
        return (1, 1, input_shape[0] * input_shape[1] * input_shape[2])

    def compute_delta(self, delta: np.array):
        print(f"delta.reshape(self.shape_before): {delta.reshape(self.shape_before)}")
        return delta.reshape(self.shape_before)

    def update_weights(self, learning_rate: float):
        pass
