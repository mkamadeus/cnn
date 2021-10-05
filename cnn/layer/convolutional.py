from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes, add_all_feature_maps
import numpy as np
from typing import Tuple
from icecream import ic


class Convolutional(BaseLayer):
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
        self.inputs = None
        self.outputs = None

        self.delta_filter = []

        # uniformly create a 4D random matrix based on kernel shape if no kernel is supplied
        # with shape of (n_channels, n_filter, w_kernel_shape, h_kernel_shape)
        if filters is None:
            self.filters = np.array(
                generate_random_uniform_matrixes(self.filter_count, self.n_channels, self.kernel_shape)
            )
        else:
            self.filters = filters

        # TODO: bias confirm
        self.bias = 1
        self.bias_weight = 0
        self.type = "convolutional"

        self.delta = 0

        ic(self.input_shape)

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
                # ic(multiplied_views)
                # apply convolutional multiplication
                conv_mult_res = np.array([[np.sum(view) for view in row] for row in multiplied_views])

                # save convolution multiplication to channel feature map
                feature_map.append(conv_mult_res)

            # convert to np.array
            feature_map = np.array(feature_map)
            ic(feature_map)

            # Add all channel feature maps and then store on final feature
            # maps array
            final_feature_maps.append(add_all_feature_maps(feature_map))
            ic(final_feature_maps)

            # increment filter index to move to the next filter
            filter_idx += 1

        bias_weight = self.bias * self.bias_weight
        return np.array(final_feature_maps) + bias_weight

    def run(self, inputs: np.array):
        # Handling error of input
        # If number of channels of input is inequal
        ic(inputs, self.kernel_shape)
        if inputs.shape != self.input_shape:
            raise ValueError(f"input shape mismatch, found {inputs.shape} should be {self.input_shape}.")

        self.inputs = inputs
        self.outputs = self.run_convolution_stage(inputs)

        return self.outputs

    def compute_delta(self, delta: np.ndarray):
        final_delta_filters = []
        filter_idx = 0

        ic(delta)

        for d in delta:
            delta_filters = []
            for channel_idx, input_channel in enumerate(self.inputs):
                # setup input with padding
                padded = pad_array(input_channel, self.padding, 0)

                # aka receptive fields
                strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)

                multiplied_views = np.array([np.multiply(view, d) for view in strided_views])

                # apply convolutional multiplication
                self.delta_filter = np.array([[np.sum(view) for view in row] for row in multiplied_views])

                ic(self.delta_filter)

                # save convolution multiplication to channel feature map
                delta_filters.append(self.delta_filter)

            # convert to np.array
            delta_filters = np.array(delta_filters)
            ic(delta_filters)

            # Add all channel feature maps and then store on final feature
            # maps array
            final_delta_filters.append(add_all_feature_maps(delta_filters))
            ic(final_delta_filters)

            # increment filter index to move to the next filter
            filter_idx += 1

        final_delta_filters = np.array(final_delta_filters)
        self.delta += final_delta_filters

        # TODO: backprop fix

        return final_delta_filters

    def update_weight(self, learning_rate: float):
        self.filters = self.filters - learning_rate * self.delta
        self.delta = 0

    def get_shape(self):
        length = (self.input_shape[1] + 2 * self.padding - self.kernel_shape[0]) // self.stride + 1
        width = (self.input_shape[2] + 2 * self.padding - self.kernel_shape[1]) // self.stride + 1

        return (self.filter_count, length, width)

    def get_weight_count(self):
        nb_feature_map = len(self.filters)
        nb_channel = len(self.filters[0])
        return nb_feature_map * ((nb_channel * self.kernel_shape[0] * self.kernel_shape[1]) + 1)
