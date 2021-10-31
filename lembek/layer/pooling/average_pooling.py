import numpy as np
from lembek.utils import generate_strides
from typing import Tuple
from lembek.layer.base import BaseLayer
from icecream import ic


class AveragePooling(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, stride):
        self.size: Tuple[int, int] = size
        self.stride: int = stride
        self.type = "avgpool"

    def run_pooling(self, inputs):
        result = []
        for i in inputs:
            # setup input
            strided_views = generate_strides(i, self.size, stride=self.stride)

            # make feature map
            feature_map = np.array([[np.average(view) for view in row] for row in strided_views])

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

        result = np.zeros(self.input.shape)
        for idx, err in enumerate(self.input):
            for i in range(delta[idx].shape[0]):
                if i % self.stride == 0:
                    for j in range(delta[idx].shape[0]):
                        if j % self.stride == 0:
                            input_view = np.zeros(self.size)
                            input_view.fill(delta[idx, i, j] / self.input.size)
                            ic(input_view, delta[idx])
                            result[idx, i : i + self.size[0], j : j + self.size[1]] += input_view

        ic(result)
        return result

    def update_weights(self, learning_rate: float, momentum: float):
        pass
