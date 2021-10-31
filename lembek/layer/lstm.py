import numpy as np
from lembek.activations import (
    sigmoid,
    tanh,
)
from lembek.layer.base import BaseLayer
from lembek.utils import generate_random_uniform_matrixes_lstm
from icecream import ic


class LSTM(BaseLayer):
    """
    Defines a LSTM layer.
    """

    def __init__(self, size: int, input_size: tuple, recurrent_weights=None, into_weights=None):
        self.size = size

        if len(input_size) != 2:
            raise ValueError(f"The input size should be defined on 2D shape. Found {len(input_size)}D shape.")

        self.input_size = input_size
        self.n_features = input_size[1]
        self.init_random_weight()

    def init_random_weight(self):
        # divided into 4 slots, 0: forget gate, 1: input gate, 2: cell state, 3: output gate

        # init weights (U)
        self.weights = generate_random_uniform_matrixes_lstm((4, self.size, self.n_features))

        # init recurrent weights (W)
        self.recurrent_weights = generate_random_uniform_matrixes_lstm((4, self.size, self.n_features))

        # init biases (b)
        self.biases = generate_random_uniform_matrixes_lstm((4, self.size, 1))

        # init states
        self.cell_state = np.zeros((self.size, 1))
        self.hidden_state = np.zeros((self.size, 1))

        # for testing purpose only
        # komen aja ntar
        self.weights = np.array([[[0.7, 0.45]], [[0.95, 0.8]], [[0.45, 0.25]], [[0.6, 0.4]]])

        self.recurrent_weights = np.array([[[0.1]], [[0.8]], [[0.15]], [[0.25]]])

        self.biases = np.array([[[0.15]], [[0.65]], [[0.2]], [[0.1]]])

        self.hidden_state = np.array([[0]])
        self.cell_state = np.array([[0]])

    def run(self, inputs: np.array) -> np.ndarray:
        # precalculations  (self.hidden_state is previous hidden state)

        # iterasi tiap input = iterasi tiap timestep

        for inp in inputs:
            ic(self.weights)
            ic(self.recurrent_weights)
            ic(self.hidden_state)
            # ufx_wfh = np.matmul(self.forget_weight, inputs) + np.matmul(self.forget_recurrent_weight, self.hidden_state)
            wh = np.matmul(self.hidden_state, self.recurrent_weights)
            ic(wh)
            ic(inp)
            # should reshape input because this input will be distributed to all gates
            ux_wh = np.add(np.matmul(self.weights, inp.T.reshape(-1, 1)), wh)
            ic(np.matmul(self.weights, inp.T.reshape(-1, 1)))
            ic(ux_wh)

            # ufx + wfh + bias
            nets = np.add(ux_wh, self.biases)
            ic(nets)

            gates = []
            for idx, net in enumerate(nets):
                if idx == 2:
                    gates.append(tanh(net))
                elif idx != 2:
                    gates.append(sigmoid(net))

            gates = np.array(gates)
            ic(gates)
            self.cell_state = gates[0] * self.cell_state + gates[1] * gates[2]

            ic(self.cell_state)

            self.hidden_state = gates[3] * tanh(self.cell_state)
            ic(self.hidden_state)
            # self.cell_state = self.forget_gate * self.cell_state + cell_tmp_state

        # uix_wih = np.matmul(self.input_weight, inputs) + np.matmul(self.input_recurrent_weight, self.hidden_state)
        # ucx_wch = np.matmul(self.cell_weight, inputs) + np.matmul(self.cell_recurrent_weight, self.hidden_state)
        # uox_woh = np.matmul(self.output_weight, inputs) + np.matmul(self.output_recurrent_weight, self.hidden_state)

        # # forget gate
        # self.forget_gate = sigmoid(ufx_wfh + self.forget_bias)

        # # input gate
        # self.input_gate = sigmoid(uix_wih + self.input_bias)
        # # self.input_gate = sigmoid(self.input_gate)

        # # cell state (self.cell_state is previous cell state, being set to current cell state)
        # cell_tmp_state = tanh(ucx_wch + self.cell_bias)
        # cell_tmp_state = cell_tmp_state * self.input_gate
        # self.cell_state = self.forget_gate * self.cell_state + cell_tmp_state

        # # output gate
        # self.output_state = sigmoid(uox_woh + self.output_bias)

        # # set hidden state (self.hidden_state being set to current hidden state)
        # self.hidden_state = self.output_state * tanh(self.cell_state)

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
