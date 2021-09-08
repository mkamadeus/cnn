import numpy as np
from cnn.utils import generate_strides, pad_array
from typing import Tuple
from cnn.layer.base import BaseLayer
from icecream import ic

POOLING_MODES = ["max", "average"]


class PoolingLayer(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, stride, mode="average"):
        if mode not in POOLING_MODES:
            raise Exception("invalid pooling mode")

        self.size: Tuple[int, int] = size
        self.stride: int = stride
        self.mode: str = mode

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        # setup input
        strided_views = generate_strides(inputs, self.size, stride=self.stride)
        ic(strided_views)

        # make feature map
        if self.mode == "average":
            feature_map = np.array([[np.average(view) for view in row] for row in strided_views])
        elif self.mode == "max":
            feature_map = np.array([[np.max(view) for view in row] for row in strided_views])

        ic(feature_map)

        return feature_map
