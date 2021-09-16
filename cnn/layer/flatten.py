import numpy as np

# from cnn.utils import generate_strides, pad_array
from typing import Tuple
from cnn.layer.base import BaseLayer


class Flatten(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self, size):
        self.size: Tuple[int, int] = size
        self.type = "flatten      "

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        flatten_output = inputs.flatten()
        return flatten_output

    def get_shape(self, input_shape=None):
        input_shape = super().get_shape(input_shape)
        return (1, 1, input_shape[0] * input_shape[1] * input_shape[2])
