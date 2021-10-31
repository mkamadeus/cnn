from lembek.layer.base import BaseLayer
from lembek.utils import generate_strides, pad_array, generate_random_uniform_matrixes, add_all_feature_maps
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
        self.n_channels = input_shape[0]
        self.inputs = None
        self.outputs = None

        self.velocity = 0

        self.delta_filter = []

        # uniformly create a 4D random matrix based on kernel shape if no kernel is supplied
        # with shape of (n_filter, n_channels, w_kernel_shape, h_kernel_shape)
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
                ic(kernels.shape)
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
        # print(f"delta shape: {delta.shape}")

        for d in delta:
            delta_filters = []
            for channel_idx, input_channel in enumerate(self.inputs):
                # print(f"input_channel shape: {input_channel.shape}")
                # setup input with padding
                padded = pad_array(input_channel, self.padding, 0)
                # print(f"padded shape: {padded.shape}")

                # aka receptive fields
                strided_views = generate_strides(padded, d.shape, stride=self.stride)
                # print(f"strided views shape: {strided_views.shape}")

                multiplied_views = np.array([np.multiply(view, d) for view in strided_views])

                # apply convolutional multiplication
                self.delta_filter = np.array([[np.sum(view) for view in row] for row in multiplied_views])

                ic(self.delta_filter)

                # save convolution multiplication to channel feature map
                delta_filters.append(self.delta_filter)

            # convert to np.array
            delta_filters = np.array(delta_filters)
            ic(delta_filters)

            # Append the delta filters for this filter
            final_delta_filters.append(delta_filters)
            ic(final_delta_filters)

            # increment filter index to move to the next filter
            filter_idx += 1

        final_delta_filters = np.array(final_delta_filters)
        self.delta += final_delta_filters

        ic(final_delta_filters)
        # ic(self.filters)
        # ic(self.filters.shape)
        # ic(np.rot90(self.filters,k=2, axes=(0,1)))
        ic(np.rot90(self.filters, k=2, axes=(-2, -1)))
        rotated_filters = np.rot90(self.filters, k=2, axes=(-2, -1))
        # print(f"self.filters.shape: {self.filters.shape}")
        # print(f"rotated_filters[1]: {rotated_filters[1]}")
        # print(final_delta_filters)

        conv_delta = []
        filter_idx = 0
        # ic(rotated_filters)

        # for every delta
        for delta_idx, input_delta in enumerate(delta):
            delta_inputs = []
            # for every rotated filter on certain filter index
            for r in rotated_filters[filter_idx]:
                # pad the delta
                padded = pad_array(input_delta, self.filters.shape[-1] - 1, 0)
                # obtain receptive fields
                strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
                # multiply the receptive fields with rotated filter
                multiplied_views = np.array([np.multiply(view, r) for view in strided_views])
                ic(multiplied_views)
                # apply convolutional multiplication
                conv_mult_res = np.array([[np.sum(view) for view in row] for row in multiplied_views])

                # save convolution multiplication to delta inputs array on this filter
                delta_inputs.append(conv_mult_res)

            # convert to np.array
            delta_inputs = np.array(delta_inputs)
            ic(delta_inputs)

            # Append delta input for this filter
            conv_delta.append(delta_inputs)
            # ic(conv_delta)

            # increment filter index to move to the next filter
            filter_idx += 1
        conv_delta = np.array(conv_delta)
        ic(sum(conv_delta))

        # So this is element wise addition of all delta inputs, for all delta inputs
        # within the same channel. (Inversed version of element wise addition for forward prop
        # where within the same filter)
        return sum(conv_delta)

    def get_type(self):
        return "conv2d"

    def update_weights(self, learning_rate: float, momentum: float):
        # print(f"filters shape: {self.filters.shape}")
        # print(f"delta shape: {self.delta.shape}")
        self.filters = self.filters - learning_rate * self.delta + momentum * self.velocity
        self.velocity = self.delta
        self.delta = 0

    def get_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape

        length = (self.input_shape[1] + 2 * self.padding - self.kernel_shape[0]) // self.stride + 1
        width = (self.input_shape[2] + 2 * self.padding - self.kernel_shape[1]) // self.stride + 1

        return (self.filter_count, length, width)

    def get_weight_count(self):
        nb_feature_map = len(self.filters)
        nb_channel = self.input_shape[0]
        return nb_feature_map * (nb_channel * self.kernel_shape[0] * self.kernel_shape[1] + 1)
