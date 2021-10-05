import numpy as np
from cnn.utils import generate_strides
from typing import Tuple
from cnn.layer.base import BaseLayer
from icecream import ic


def max_pooling_mask(view: np.ndarray, value: float, delta_value: float):
    result = np.array([[delta_value if col == value else 0 for col in row] for row in view])
    return result


class MaxPooling(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, stride):
        self.size: Tuple[int, int] = size
        self.stride: int = stride
        self.type = "pooling      "

    def run_pooling(self, inputs):
        result = []
        for i in inputs:
            # setup input
            strided_views = generate_strides(i, self.size, stride=self.stride)

            # make feature map
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

        result = np.zeros(self.input.shape)
        for idx in range(len(self.input)):
            for i in range(delta[idx].shape[0]):
                if i % self.stride == 0:
                    for j in range(delta[idx].shape[0]):
                        if j % self.stride == 0:
                            max_value = self.output[idx, i, j]
                            err_value = delta[idx, i, j]
                            ic(max_value, err_value)

                            input_view = np.zeros(self.size).flatten()
                            for k, input_value in enumerate(
                                self.input[idx, i : i + self.size[0], j : j + self.size[1]].flatten()
                            ):
                                if input_value == max_value:
                                    input_view[k] = err_value
                                    break
                            input_view = input_view.reshape(self.size)
                            ic(input_view)

                            result[idx, i : i + self.size[0], j : j + self.size[1]] += input_view

        ic(result)
        return result

    def update_weight(self):
        pass
