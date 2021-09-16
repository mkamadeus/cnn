import numpy as np
from cnn.utils import generate_strides
from typing import Tuple
from cnn.layer.base import BaseLayer

POOLING_MODES = ["max", "average"]


class Pooling(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, stride, mode="average"):
        if mode not in POOLING_MODES:
            raise Exception("invalid pooling mode")

        self.size: Tuple[int, int] = size
        self.stride: int = stride
        self.mode: str = mode

    def run_pooling(self, inputs):
        result = []
        for i in inputs:
            # setup input
            strided_views = generate_strides(i, self.size, stride=self.stride)

            # make feature map
            if self.mode == "average":
                feature_map = np.array([[np.average(view) for view in row] for row in strided_views])
            elif self.mode == "max":
                feature_map = np.array([[np.max(view) for view in row] for row in strided_views])

            result.append(feature_map)
        return np.array(result)

    def run(self, inputs: np.array):
        res = self.run_pooling(inputs)
        return res
