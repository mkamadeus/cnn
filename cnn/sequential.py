from typing import List
from cnn.errors import mean_sum_squared_error
from cnn.layer.base import BaseLayer
from icecream import ic
import numpy as np
from tqdm import tqdm
import pickle

from cnn.layer.output import Output


class Sequential:
    """
    🥒 Sequential model class definition.
    """

    def __init__(
        self,
        layers: List[BaseLayer] = None,
        learning_rate: float = 0.5,
        epoch: int = 10,
    ):
        self.type = "Sequential"
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

        self.learning_rate = learning_rate
        self.epoch = epoch

    def add(self, layer: BaseLayer):
        """
        🥒 Adds a new layer to the sequential model.
        """
        self.layers.append(layer)

    def stochastic_run(self, inputs: np.ndarray, targets: np.ndarray):
        """
        🥒 Runs a stochastic process (update per one input)
        """
        if len(inputs) != len(targets):
            raise ValueError("input count and target count not equal")

        for _ in tqdm(range(self.epoch)):
            current_result = []
            for target, input_data in list(zip(targets, inputs)):
                r = self.forward_phase(input_data)
                self.backward_phase(target)
                self.update_weights()
                current_result.append(r)

            current_result = np.array(current_result)
            print(f"Error : {mean_sum_squared_error(current_result, targets)}")

    def batch_run(self, inputs: np.ndarray, targets: np.ndarray):
        """
        🥒 Runs a batch process (update per one batch/all input)
        """
        if len(inputs) != len(targets):
            raise ValueError("input count and target count not equal")

        for _ in tqdm(range(self.epoch)):
            for target, input_data in list(zip(targets, inputs)):
                self.forward_phase(input_data)
                self.backward_phase(target)
            self.update_weights()

    def mini_batch_run(self, inputs: np.ndarray, targets: np.ndarray, batch_size=5):
        """
        🥒 Runs a mini-batch process (update per batch size)
        """
        if len(inputs) != len(targets):
            raise ValueError("input count and target count not equal")

        for _ in tqdm(range(self.epoch)):
            for idx, (target, input_data) in enumerate(list(zip(targets, inputs))):
                if idx % batch_size == 0:
                    self.update_weights()
                self.forward_phase(input_data)
                self.backward_phase(target)
            self.update_weights()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        🥒 Does a prediction using current weights with supplied input and target.
        """
        result = []

        for input_data in inputs:
            # forward propagation
            self.forward_phase(input_data)

            # get result
            if type(self.layers[-1]) != Output:
                raise TypeError("last layer should be output layer")

            prediction = self.layers[-1].predict()
            result.append(prediction)

        return np.array(result)

    # TODO: soon to be deprecated
    def run(self, inputs):
        final_result = []

        for i in tqdm(inputs):
            result = i
            for idx, layer in enumerate(self.layers):
                ic(idx, result.shape, result)
                result = layer.run(result)
            final_result.append(result)

        return np.array(final_result)

    def summary(self, input_shape=None):
        """
        🥒 Output summary.
        """
        if input_shape is None:
            n_channel = len(self.inputs[0])
            length = len(self.inputs[0][0])
            width = len(self.inputs[0][0][0])
            input_shape = (n_channel, length, width)
        total_weight = 0
        print(f"Model: {self.type}")
        print("----------------------------------------------------")
        print("Layer (type)\t\tOutput Shape\t\tParam")
        print("====================================================")
        for index, layer in enumerate(self.layers):
            layer_type = layer.get_type()
            layer_shape, layer_weight = layer.get_shape_and_weight_count(input_shape)
            total_weight += layer_weight
            print(f"{layer_type}\t\t{layer_shape}\t\t{layer_weight}")
            if index != len(self.layers) - 1:
                print("----------------------------------------------------")
            input_shape = layer_shape
        print("====================================================")
        print(f"Total param/weight: {total_weight}")

    def forward_phase(self, input_data: np.ndarray):
        """
        🥒 Forward propagation.
        """
        current_output = input_data
        for idx, layer in enumerate(self.layers):
            ic(idx)
            current_output = layer.run(current_output)
        return current_output

    def backward_phase(self, target: np.ndarray):
        """
        🥒 Backward propagation.
        """
        current_delta = target
        for idx, layer in enumerate(reversed(self.layers)):
            ic(idx)
            current_delta = layer.compute_delta(current_delta)

        return current_delta

    def update_weights(self):
        """
        🥒 Update trainable parameters.
        """
        for idx, layer in enumerate(self.layers):
            ic(idx)
            layer.update_weights(self.learning_rate)

    def save(self, filename: str = "model"):
        """
        🥒 Pickle this model.
        """
        opened_file = open(f"{filename}.picl", "wb")
        pickle.dump(self, opened_file)
        opened_file.close()
