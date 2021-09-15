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
        self.bias = 1
        self.bias_weight = 1

        # uniformly create a 4D random matrix based on kernel shape
        # with shape of (n_channels, n_filter, w_kernel_shape, h_kernel_shape)
        self.kernels = generate_random_uniform_matrixes(self.filter_count, self.n_channels, self.kernel_shape)
        ic(self.kernels)
        ic(self.kernels.shape)

    def run_convolution_stage(self, inputs: np.array):
        final_feature_maps = []
        filter_idx = 0

        for filter_kernels in self.kernels:
            filter_feature_map = []
            channel_idx = 0
            ic(filter_kernels)

            for input_channel in inputs:
                # setup input
                padded = pad_array(input_channel, self.padding, 0)
                ic(padded)

                # a. k. a receptive fields
                strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
                ic(strided_views)

                multiplied_views = np.array([np.multiply(view, filter_kernels[channel_idx]) for view in strided_views])
                ic(multiplied_views)

                # apply convolutional multiplication
                conv_mult_res = np.array([[np.sum(view) for view in row] for row in multiplied_views])
                ic(conv_mult_res)

                # save convolution multiplication to channel feature map
                filter_feature_map.append(conv_mult_res)

                channel_idx += 1

            ic(filter_feature_map)
            # Add all channel feature maps and then store on final feature
            # maps array
            final_feature_maps.append(add_all_feature_maps(np.array(filter_feature_map)))
            ic(final_feature_maps)
            # increment filter index to move to the next filter
            filter_idx += 1

        bias_weight = self.bias * self.bias_weight
        return np.array(final_feature_maps)+bias_weight

    # TODO: adjust with pooling
    def run(self, inputs: np.array):
        # Handling error of input
        # If number of channels of input is inequal
        if (inputs.shape != self.input_shape):
            raise ValueError(f'The input shape is invalid. It should be {self.input_shape}.')

        return self.run_convolution_stage(inputs)
