from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes, add_all_feature_maps
import numpy as np
from typing import Tuple
from icecream import ic


class ConvolutionalLayer(BaseLayer):
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        padding: int,
        filter_count: int,
        kernel_shape: Tuple[int, int],
        stride: int,
        filters: np.array = None,
    ):
        if len(input_shape) != 3:
            raise TypeError("the input shape should be 3D")
        if len(kernel_shape) != 2:
            raise TypeError("the kernel shape should be 2D")
        if padding < 0:
            raise ValueError("padding should be >= 0")
        if filter_count < 1:
            raise ValueError("filter count should be >= 1")
        if stride < 1:
            raise ValueError("stride should be >= 1")

        self.input_shape = input_shape
        self.padding = padding
        self.stride = stride
        self.filter_count = filter_count
        self.kernel_shape = kernel_shape
        self.n_channels = input_shape[1]

        # uniformly create a 4D random matrix based on kernel shape if no kernel is supplied
        # with shape of (n_channels, n_filter, w_kernel_shape, h_kernel_shape)
        if filters is None:
            self.filters = np.array(generate_random_uniform_matrixes(self.filter_count, self.n_channels, self.kernel_shape))
        else:
            self.filters = filters
        ic(self.filters.shape)

        # TODO: bias confirm
        self.bias = 1
        self.bias_weight = 0

    def run_convolution_stage(self, inputs: np.array):
        final_feature_maps = []
        filter_idx = 0

        for kernels in self.filters:
            feature_map = []

            for channel_idx, input_channel in enumerate(inputs):
                # setup input with padding
                padded = pad_array(input_channel, self.padding, 0)

                # aka receptive fields
                strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
                multiplied_views = np.array([np.multiply(view, kernels[channel_idx]) for view in strided_views])

                # apply convolutional multiplication
                conv_mult_res = np.array([[np.sum(view) for view in row] for row in multiplied_views])

                # save convolution multiplication to channel feature map
                feature_map.append(conv_mult_res)

                channel_idx += 1

            # convert to np.array
            feature_map = np.array(feature_map)
            ic(feature_map)

            # Add all channel feature maps and then store on final feature
            # maps array
            final_feature_maps.append(add_all_feature_maps(feature_map))

            # increment filter index to move to the next filter
            filter_idx += 1

        bias_weight = self.bias * self.bias_weight
        return np.array(final_feature_maps) + bias_weight

    # TODO: adjust with pooling
    def run(self, inputs: np.array):
        # Handling error of input
        # If number of channels of input is inequal
        if inputs.shape != self.input_shape:
            raise ValueError(f"input shape mismatch, found {inputs.shape} should be {self.input_shape}.")

        return self.run_convolution_stage(inputs)
