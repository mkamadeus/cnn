import numpy as np
from cnn.layer.base import BaseLayer


class Flatten(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self):
        pass

    def run(self, inputs: np.array):
        flatten_output = inputs.flatten()
        return flatten_output
