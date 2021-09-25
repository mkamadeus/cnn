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

    def run(self, inputs: np.array) -> np.ndarray:
        if len(inputs.shape) != 1:
            raise ValueError("input data should be 1D")

        # save input for backprop
        self.input = inputs

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

        # save activated output for backprop
        self.output = result
        return result

    def get_shape(self, input_shape=None):
        return (1, 1, self.size)

    def get_weight_count(self):
        return len(self.weights.flatten())

    def compute_delta(self, delta: np.ndarray):
        # store deltas for bias and weight
        self.delta_bias = delta
        self.delta_weight = np.matmul(np.array([self.output]).T, np.array([delta]))

        ic(delta, self.output.T)
        ic(self.delta_bias, self.delta_weight)

        # calculate delta for prev layer
        delta_activation = np.matmul(np.array([delta]), self.weights[1:].T)
        ic(delta_activation)

        # TODO: verify the truthiness of this formula.. not really sure lol
        # set derivative of activation function
        if self.activation == "relu":
            derivative_activation_function = relu_derivative
        elif self.activation == "sigmoid":
            derivative_activation_function = sigmoid_derivative
        # delta_layer = np.ma.array(data=delta_activation, mask=~(self.input > 0), fill_value=0).filled()\
        delta_layer = delta_activation * derivative_activation_function(self.input)
        ic(delta_layer)
        return delta_layer.flatten()

    def update_weight(self):
        self.weights -= self.learning_rate * self.delta
        return self.weights
