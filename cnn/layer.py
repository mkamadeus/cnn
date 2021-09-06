import numpy as np


class ConvolutionalLayer:
    """
    Defines a convolutional layer consisting of inputs and kernels.
    """

    def __init__(self, inputs, kernels):
        self.inputs = inputs
        self.kernels = kernels

    def forward_propagation(self):
        """
        Does a CNN forward propagation.
        """
        self.convolution()
        self.detector()
        self.pooling()

    def convolution(self, stride: int, padding: int):
        pass

    def detector(self):
        pass

    def pooling(self):
        pass
