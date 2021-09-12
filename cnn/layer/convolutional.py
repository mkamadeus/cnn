from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes, add_all_feature_maps
import numpy as np
from icecream import ic
from typing import Tuple


class ConvolutionalLayer(BaseLayer):
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(self, input_shape: Tuple[int, int, int], padding: int, filter_count: int, kernel_shape: Tuple[int, int], stride: int):
        if (len(input_shape) != 3):
            raise TypeError('The input shape should be on 3D, which means the tuple should consists of 3 values.')
        self.input_shape = input_shape
        self.padding = padding
        self.filter_count = filter_count
        if (len(kernel_shape) != 2):
            raise TypeError('The kernel shape should be 2D, which means the tuple should consists of 2 values.')
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.n_channels = input_shape[0]

        # uniformly create a 4D random matrix based on kernel shape
        # with shape of (n_channels, n_filter, w_kernel_shape, h_kernel_shape)
        self.kernels = generate_random_uniform_matrixes(self.filter_count, self.n_channels, self.kernel_shape)
        ic(self.kernels)

    def run_convolution_stage(self, inputs: np.array):
        final_feature_maps = []
        channel_num = 0
        for input_channel in inputs:
            # setup input
            padded = pad_array(input_channel, self.padding, 1)
            ic(padded)

            # a. k. a receptive fields
            strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
            ic(strided_views)

            # for channel_kernels in self.kernels:
            channel_feature_map = []

            for kernel in self.kernels[channel_num]:
                multiplied_views = np.array([np.multiply(view, kernel) for view in strided_views])
                ic(multiplied_views)
                # ic(multiplied_views.shape)

                # make feature map
                conv_mult_res = np.array([[np.sum(view) for view in row] for row in multiplied_views])
                ic(conv_mult_res)

                # save convolution multiplication to channel feature map
                channel_feature_map.append(conv_mult_res)

            # Add all channel feature maps and then store on final feature
            # maps array
            final_feature_maps.append(add_all_feature_maps(np.array(channel_feature_map)))

            # increment channel num to move to next channel
            channel_num += 1

        return np.array(final_feature_maps)

    # TODO: adjust with pooling
    def run(self, inputs: np.array):
        # Handling error of input
        # If number of channels of input is inequal
        if (inputs.shape != self.input_shape):
            raise ValueError(f'The input shape is invalid. It should be {self.input_shape}.')

        return self.run_convolution_stage(inputs)
