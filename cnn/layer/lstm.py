import numpy as np
from cnn.activations import linear, relu_derivative, sigmoid, relu, sigmoid_derivative, softmax, linear_derivative, tanh
from cnn.layer.base import BaseLayer
from cnn.utils import generate_random_uniform_matrixes_lstm
from icecream import ic


class LSTM(BaseLayer):
    """
    Defines a LSTM layer.
    """

    def __init__(self, size: int, input_size: tuple, recurrent_weights=None, into_weights=None):
        self.size = size

        if len(input_size) != 2:
            raise ValueError(f"The input size should be defined on 2D shape. Found {len(input_size)}D shape.")
        if recurrent_weights is None:
            # init random recurrent weights
            pass
        else:
            # ngeassign satu-satu ke weightnya
            pass
        if into_weights is None:
            # init random U weights
            pass
        else:
            # ngeassign satu-satu ke weightnya
            pass

        self.input_size = input_size
        self.n_features = input_size[1]
        self.init_random_weight()

    def init_random_weight(self):
        # initialize forget gate related stuff (h)
        self.forget_weight = generate_random_uniform_matrixes_lstm((self.size, self.n_features))
        self.forget_recurrent_weight = generate_random_uniform_matrixes_lstm((self.size, self.size))
        # kayaknya ga dipake
        # self.forget_state = generate_random_uniform_matrixes_lstm(self.input_size)
        self.forget_bias = generate_random_uniform_matrixes_lstm((self.size, 1))

        # initialize input gate related stuff(i)
        self.input_weight = generate_random_uniform_matrixes_lstm((self.size, self.n_features))
        self.input_recurrent_weight = generate_random_uniform_matrixes_lstm((self.size, self.size))
        # kayaknya ga dipake
        # self.input_state = generate_random_uniform_matrixes_lstm(self.input_size)
        self.input_bias = generate_random_uniform_matrixes_lstm((self.size, 1))

        # initialize cell state
        self.cell_weight = generate_random_uniform_matrixes_lstm((self.size, self.n_features))
        self.cell_recurrent_weight = generate_random_uniform_matrixes_lstm((self.size, self.size))
        self.cell_state = np.zeros((self.size, 1))
        self.hidden_state = np.zeros((self.size, 1))
        self.cell_bias = generate_random_uniform_matrixes_lstm((self.size, 1))

        # initialize cell state
        self.output_weight = generate_random_uniform_matrixes_lstm((self.size, self.n_features))
        self.output_recurrent_weight = generate_random_uniform_matrixes_lstm((self.size, self.size))
        self.output_state = np.zeros((self.size, 1))
        self.output_bias = generate_random_uniform_matrixes_lstm((self.size, 1))

    def run(self, inputs: np.array) -> np.ndarray:
        # precalculations  (self.hidden_state is previous hidden state)
        ufx_wfh = np.matmul(self.forget_weight, inputs) + np.matmul(self.forget_recurrent_weight, self.hidden_state)
        ic(ufx_wfh)
        uix_wih = np.matmul(self.input_weight, inputs) + np.matmul(self.input_recurrent_weight, self.hidden_state)
        ucx_wch = np.matmul(self.cell_weight, inputs) + np.matmul(self.cell_recurrent_weight, self.hidden_state)
        uox_woh = np.matmul(self.output_weight, inputs) + np.matmul(self.output_recurrent_weight, self.hidden_state)

        # forget gate
        self.forget_gate = sigmoid(ufx_wfh + self.forget_bias)

        # input gate
        self.input_gate = sigmoid(uix_wih + self.input_bias)
        # self.input_gate = sigmoid(self.input_gate)

        # cell state (self.cell_state is previous cell state, being set to current cell state)
        cell_tmp_state = tanh(ucx_wch + self.cell_bias)
        cell_tmp_state = cell_tmp_state * self.input_gate
        self.cell_state = self.forget_gate * self.cell_state + cell_tmp_state

        # output gate
        self.output_state = sigmoid(uox_woh + self.output_bias)

        # set hidden state (self.hidden_state being set to current hidden state)
        self.hidden_state = self.output_state * tanh(self.cell_state)

        return self.hidden_state

    def compute_delta(self, delta: np.ndarray):
        pass

    def update_weights(self, learning_rate: float, momentum: float):
        pass

    def get_type(self):
        return "lstm"

    def get_shape(self, input_shape=None):
        return (1, 1, self.size)

    def get_weight_count(self):
        return 4 * self.size * (self.n_features + self.size + 1)
