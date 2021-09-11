from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes, add_all_feature_maps
import numpy as np
from icecream import ic


class ConvolutionalLayer(BaseLayer):
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(self, input_shape: tuple, padding: int, filter_count: int, kernel_shape: tuple, stride: int):
        self.input_shape = input_shape
        self.padding = padding
        self.filter_count = filter_count
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.n_channels = input_shape[0]

        # uniformly channel*filter_count random matrix based on kernel shape
        self.kernels = generate_random_uniform_matrixes(self.filter_count, self.n_channels, self.kernel_shape)
        ic(self.kernels)

    def run_convolution_stage(self, inputs: np.array):
        # setup input
        padded = pad_array(inputs, self.padding, 0)

        # a. k. a receptive fields
        strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
        ic(strided_views)

        final_feature_maps = []
        for channel_kernels in self.kernels:
            channel_feature_map = []

            for kernel in channel_kernels:
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

        return np.array(final_feature_maps)

    # TODO: adjust with pooling
    def run(self, inputs: np.array):
        return self.run_convolution_stage(inputs)
