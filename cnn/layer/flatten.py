import numpy as np
# from cnn.utils import generate_strides, pad_array
from typing import Tuple
from cnn.layer.base import BaseLayer
from icecream import ic


class FlattenLayer(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self, size):
        self.size: Tuple[int, int] = size

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        flatten_output = inputs.flatten()
        ic(flatten_output)
        return flatten_output
