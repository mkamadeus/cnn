from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array
import numpy as np
from icecream import ic


class ConvolutionalLayer(BaseLayer):
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(self, kernel: np.array, stride: int, padding: int):
        self.stride = stride
        self.padding = padding
        self.kernel = kernel

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        # setup input
        padded = pad_array(inputs, self.padding, 0)
        strided_views = generate_strides(padded, self.kernel.shape, stride=self.stride)
        ic(strided_views)

        # multiply all view with kernel
        multiplied_views = np.array([np.multiply(view, self.kernel) for view in strided_views])
        ic(multiplied_views)
        ic(multiplied_views.shape)

        # make feature map
        feature_map = np.array([[np.sum(view) for view in row] for row in multiplied_views])
        ic(feature_map)

        return feature_map
