import numpy as np
from cnn.utils import generate_strides
from typing import Tuple
from cnn.layer.base import BaseLayer
from icecream import ic

POOLING_MODES = ["max", "average"]


def max_pooling_mask(view: np.ndarray, value: float, delta_value: float):
    result = np.array([[delta_value if col == value else 0 for col in row] for row in view])
    return result


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
        self.type = "pooling      "

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

    def get_shape(self, input_shape=None):
        input_shape = super().get_shape(input_shape)
        length = (input_shape[1] - self.size[0]) // self.stride + 1
        width = (input_shape[2] - self.size[1]) // self.stride + 1
        return (input_shape[0], length, width)

    def run(self, inputs: np.array):
        self.input = inputs
        res = self.run_pooling(inputs)
        self.output = res
        return res

    def compute_delta(self, delta: np.ndarray):
        ic(delta, self.input, self.output)

        # TODO: backprop for average
        result = []
        for idx, i in enumerate(self.input):
            strided_views = generate_strides(i, self.size, stride=self.stride)
            ic(strided_views, self.output[0, 0][0], delta[idx])

            masked = np.array(
                [
                    [max_pooling_mask(view, self.output[r, c, 0], delta[idx]) for c, view in enumerate(row)]
                    for r, row in enumerate(strided_views)
                ]
            )
            ic(masked)
            # np.ma.array(data=i, mask=~(self.input > ), fill_value=0).filled()
            result.append(masked)

        result = np.array(result)
        ic(result)

        return delta
