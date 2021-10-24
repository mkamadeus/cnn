import numpy as np
from cnn.activations import linear, relu_derivative, sigmoid, relu, sigmoid_derivative, softmax, linear_derivative, tanh
from cnn.layer.base import BaseLayer
from icecream import ic


class LSTM(BaseLayer):
    """
    Defines a LSTM layer.
    """

    def __init__(self, size, input_size):
        self.size = size
        self.input_size = input_size

        # initialize forget gate related stuff (h)
        self.forget_weight = np.zeros((input_size, size))
        self.forget_recurrent_weight = np.zeros((size, size))
        self.forget_state = np.zeros(input_size)
        self.forget_bias = np.zeros(size)

        # initialize input gate related stuff(i)
        self.input_weight = np.zeros((input_size, size))
        self.input_recurrent_weight = np.zeros((size, size))
        self.input_state = np.zeros(input_size)
        self.input_bias = np.zeros(size)

        # initialize cell state
        self.cell_weight = np.zeros((input_size, size))
        self.cell_recurrent_weight = np.zeros((size, size))
        self.cell_state = np.zeros(input_size)
        self.cell_bias = np.zeros(size)

        # initialize cell state
        self.output_weight = np.zeros((input_size, size))
        self.output_recurrent_weight = np.zeros((size, size))
        self.output_state = np.zeros(input_size)
        self.output_bias = np.zeros(size)

    def run(self, inputs: np.array) -> np.ndarray:
        # precalculations  (self.hidden_state is previous hidden state)
        ufx_wfh = np.matmul(self.forget_weight, inputs) + np.matmul(self.forget_reccurent_weight, self.hidden_state)
        uix_wih = np.matmul(self.input_weight, inputs) + np.matmul(self.input_reccurent_weight, self.hidden_state)
        ucx_wch = np.matmul(self.cell_weight, inputs) + np.matmul(self.cell_reccurent_weight, self.hidden_state)
        uox_woh = np.matmul(self.output_weight, inputs) + np.matmul(self.output_reccurent_weight, self.hidden_state)

        # forget gate
        self.forget_gate = sigmoid(ufx_wfh + self.forget_bias)

        # input gate
        self.input_gate = sigmoid(uix_wih + self.input_bias)
        self.input_gate = sigmoid(self.input_gate)

        # cell state (self.cell_state is previous cell state, being set to current cell state)
        cell_tmp_state = tanh(ucx_wch + self.cell_bias)
        cell_tmp_state = cell_tmp_state * self.input_gate
        self.cell_state = self.forget_gate * self.cell_state + cell_tmp_state

        # output gate
        self.output_state = sigmoid(uox_woh + self.output_bias)

        # set hidden state (self.hidden_state being set to current hidden state)
        self.hidden_state = self.output_state * tanh(self.cell_state)

    def compute_delta(self, delta: np.ndarray):
        pass

    def update_weights(self, learning_rate: float, momentum: float):
        pass

    def get_type(self):
        return "lstm"

    def get_shape(self, input_shape=None):
        return (1, 1, self.size)

    def get_weight_count(self):
        return 4 * (self.input_size * self.size + self.size * self.size + self.size)
