from cnn.layer.base import BaseLayer
from cnn.utils import generate_strides, pad_array, generate_random_uniform_matrixes
import numpy as np
from icecream import ic
from cnn.activations import relu


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
        self.kernels = generate_random_uniform_matrixes(self.n_channels * self.filter_count, self.kernel_shape)
        ic(self.kernels.shape)

    def run_convolution_stage(self, inputs: np.array):
        # setup input
        padded = pad_array(inputs, self.padding, 0)

        # a. k. a receptive fields
        strided_views = generate_strides(padded, self.kernel_shape, stride=self.stride)
        ic(strided_views)

        feature_maps_reg_array = []
        # TODO: sum feature map for each channel with element wise addition
        for kernel in self.kernels:
            # multiply all views with kernel
            multiplied_views = np.array([np.multiply(view, kernel) for view in strided_views])
            ic(multiplied_views)
            ic(multiplied_views.shape)

            # make feature map
            feature_map = np.array([[np.sum(view) for view in row] for row in multiplied_views])
            ic(feature_map)

            feature_maps_reg_array.append(feature_map)
        ic(np.array(feature_maps_reg_array))
        return np.array(feature_maps_reg_array)

    def detector(self, feature_map: np.array):
        # Preparing the output of the ReLU activation function.
        # print(feature_map.shape)
        relu_out = np.zeros(feature_map.shape)
        for map_num in range(feature_map.shape[-1]):
            for r in np.arange(0, feature_map.shape[0]):
                for c in np.arange(0, feature_map.shape[1]):
                    # print(relu(feature_map[r, c, map_num]))
                    # print(feature_map[r, c, map_num])
                    relu_out[r, map_num] = relu(feature_map[r, c, map_num])

        # ic(relu_out)
        return relu_out

    # TODO: multiple channels, multiple kernels
    def run(self, inputs: np.array):
        # must return result
        # # setup input
        # padded = pad_array(inputs, self.padding, 0)

        # # TODO: for each kernel
        # strided_views = generate_strides(padded, self.kernel.shape, stride=self.stride)
        # ic(strided_views)

        # # multiply all view with kernel
        # multiplied_views = np.array([np.multiply(view, self.kernel) for view in strided_views])
        # ic(multiplied_views)
        # ic(multiplied_views.shape)

        # # make feature map
        # feature_map = np.array([[np.sum(view) for view in row] for row in multiplied_views])
        # ic(feature_map)
        print("oy")
        # return feature_map
        feature_map = self.run_convolution_stage(inputs)
        output = self.detector(feature_map)

        return output
