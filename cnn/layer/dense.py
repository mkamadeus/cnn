import numpy as np
from cnn.activations import relu_derivative, sigmoid, relu, sigmoid_derivative, softmax
from cnn.layer.base import BaseLayer
from icecream import ic

ACTIVATION_MODES = ["relu", "sigmoid", "softmax"]


class Dense(BaseLayer):
    """
    Defines a pooling layer consisting of inputs and kernels.
    """

    def __init__(self, size, input_size, weights=None, activation="sigmoid", learning_rate=0.5):
        if activation not in ACTIVATION_MODES:
            raise ValueError("invalid activation mode")

        self.type = "dense"
        self.size: int = size
        self.input_size = input_size
        if weights is None:
            self.weights = np.random.random((self.input_size + 1, size))
        else:
            self.weights = weights
        self.activation = activation
        self.learning_rate = learning_rate
        self.inputs = np.array([])
        self.outputs = np.array([])

    def run(self, inputs: np.array) -> np.ndarray:
        if len(inputs.shape) != 1:
            raise ValueError("input data should be 1D")

        ic(self.weights.shape)
        ic(inputs.shape)

        # add bias to input
        biased_input: np.ndarray = np.insert(inputs, 0, 1)
        ic(inputs)
        biased_input = np.expand_dims(biased_input, axis=1)

        result = np.matmul(self.weights.T, biased_input).flatten()
        ic(result)

        if self.activation == "sigmoid":
            activation_func = sigmoid
        elif self.activation == "relu":
            activation_func = relu
        elif self.activation == "softmax":
            activation_func = softmax(result)
        else:
            raise Exception("invalid activation mode")

        result = activation_func(result)
        ic(result)

        self.outputs = result
        return result

    def get_shape(self, input_shape=None):
        return (1, 1, self.size)

    def get_weight_count(self):
        return len(self.weights.flatten())

    def compute_delta(self, target: np.ndarray):
        # calculate dE/dout
        derr_dout = self.outputs - target

        # TODO: softmax :(
        # calculate dout/dnet
        if self.activation == "sigmoid":
            activation_func_derivative = sigmoid_derivative
        elif self.activation == "relu":
            activation_func_derivative = relu_derivative
        dout_dnet = activation_func_derivative(self.outputs)

        # calculate dnet/dw
        dnet_dw = self.inputs
        dnet_dw = np.insert(dnet_dw, 0, 1)

        # calculate dE/dw
        derr_dnet = derr_dout * dout_dnet
        ic(derr_dnet)
        self.delta = np.tile(dnet_dw, (2, 1))
        ic(self.delta)
        for i, d in enumerate(derr_dnet):
            self.delta[i] *= d

        return self.delta

    def update_weight(self):
        self.weights -= self.learning_rate * self.delta
        return self.weights
