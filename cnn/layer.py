from cnn.utils import generate_strides, pad_array
import numpy as np
from icecream import ic


class ConvolutionalLayer:
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(self, channel: np.array, kernel: np.array):
        self.channel: np.array = channel
        self.kernel: np.array = kernel

    def forward_propagation(self):
        """
        Does a CNN forward propagation.
        """
        self.convolution()
        self.detector()
        self.pooling()

    # TODO: multiple channels, multiple kernels
    def convolution(self, stride: int, padding: int):
        # setup channel
        padded = pad_array(self.channel, padding, 0)
        strided_views = generate_strides(padded, self.kernel.shape)
        ic(strided_views)

        # multiply all view with kernel
        multiplied_views = np.array([np.multiply(view, self.kernel) for view in strided_views])
        ic(multiplied_views)
        ic(multiplied_views.shape)

        # make feature map
        feature_map = np.array([[np.sum(view) for view in row] for row in multiplied_views])
        ic(feature_map)

        return feature_map

    def detector(self):
        pass

    def pooling(self):
        pass
