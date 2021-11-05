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

    def __init__(self, size: int, input_size: tuple, weights=None, recurrent_weights=None, biases=None):
        self.size = size

        if len(input_size) != 2:
            raise ValueError(f"The input size should be defined on 2D shape. Found {len(input_size)}D shape.")

        self.input_size = input_size
        self.n_features = input_size[1]

        # the weights are later divided into 4 slots, 0: forget gate, 1: input gate, 2: cell state, 3: output gate
        if weights is None:
            self.init_random_weights()
        else:
            self.weights = weights

        if recurrent_weights is None:
            self.init_random_recurrent_weights()
        else:
            self.recurrent_weights = recurrent_weights
        if biases is None:
            self.init_random_biases()
        else:
            self.biases = biases
        self.init_states()

    def init_random_weights(self):
        """
        Initialize weights (U) of the LSTM.
        """
        self.weights = generate_random_uniform_matrixes_lstm((4, self.size, self.n_features))

    def init_random_recurrent_weights(self):
        """
        Initialize reccurrent weights (W) of the LSTM.
        """
        self.recurrent_weights = generate_random_uniform_matrixes_lstm((4, self.size, self.size))

    def init_random_biases(self):
        """
        Initialize biases (b) of the LSTM.
        """
        self.biases = generate_random_uniform_matrixes_lstm((4, self.size, 1))

    def init_states(self):
        """
        Initialize cell state (c) and hidden state (h) of the LSTM.
        """
        self.cell_state = np.zeros((self.size, 1))
        self.hidden_state = np.zeros((self.size, 1))

    def run(self, inputs: np.array) -> np.ndarray:
        """
        LSTM forward propagation.
        """

        # precalculations  (self.hidden_state is previous hidden state)
        # iterasi tiap input = iterasi tiap timestep

        if len(inputs) != self.input_size[0]:
            raise ValueError(f"expected {self.input_size[0]} timesteps, found {len(inputs)}")

        # reset cell n hidden state before running timesteps
        self.init_states()
        for inp in inputs:
            ic(self.weights)
            ic(self.recurrent_weights)
            ic(self.hidden_state)

            wh = np.matmul(self.recurrent_weights, self.hidden_state)
            ic(wh)
            ic(inp)
            # should reshape input because this input will be distributed to all gates
            ux_wh = np.add(np.matmul(self.weights, inp.T.reshape(-1, 1)), wh)
            ic(np.matmul(self.weights, inp.T.reshape(-1, 1)))
            ic(ux_wh)

            # ufx + wfh + bias
            ic(self.biases)
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
            ic.disable()

        return self.hidden_state.flatten()

    def compute_delta(self, delta: np.ndarray):
        pass

    def update_weights(self, learning_rate: float, momentum: float):
        pass

    def get_type(self):
        return "lstm"

    def get_shape(self, input_shape=None):
        return self.size

    def get_weight_count(self):
        return 4 * self.size * (self.n_features + self.size + 1)
