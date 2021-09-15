import numpy as np
# from cnn.utils import generate_strides, pad_array
from cnn.activations import relu, sigmoid
from cnn.layer.base import BaseLayer
from icecream import ic


class Detector(BaseLayer):
    """
    Defines a flatten layer consisting of inputs and kernels.
    """

    def __init__(self, activation: str):
        self.activation = activation

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        if self.activation == "relu":
            act_f = lambda x: relu(x)
        elif self.activation == "sigmoid":
            act_f = lambda x: sigmoid(x)
        
        act_func = np.vectorize(act_f, otypes=[np.float])
        ic(act_func(inputs))
        return act_func(inputs)
