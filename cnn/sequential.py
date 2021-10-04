from typing import List
from cnn.layer.base import BaseLayer
from icecream import ic
import numpy as np
from tqdm import tqdm


class Sequential:
    def __init__(self, layers: List[BaseLayer] = None):
        self.type = "Sequential"
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer: BaseLayer):
        """
        Adds a new layer to the sequential model.
        """
        self.layers.append(layer)

    def stochastic_run(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Runs a stochastic process (update per one input)
        """
        if len(inputs) != len(targets):
            raise ValueError("input count and target count not equal")

        for target, input_data in list(zip(targets, inputs)):
            self.forward_phase(input_data)
            self.backward_phase(target)
            self.update_parameters()

    def batch_run(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Runs a batch process (update per one batch/all input)
        """
        pass

    def mini_batch_run(self, inputs: np.ndarray, targets: np.ndarray):
        pass
        self.inputs = inputs

    def run(self, inputs):
        final_result = []

        for i in tqdm(inputs):
            result = i
            for idx, layer in enumerate(self.layers):
                ic(idx, result.shape, result)
                result = layer.run(result)
            final_result.append(result)

        return np.array(final_result)

    def ll(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        for i in range(0, predicted.shape[0]):
            predicted[i] = min(max(1e-15, predicted[i]), 1 - 1e-15)
        err = np.seterr(all="ignore")
        score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        np.seterr(
            divide=err["divide"],
            over=err["over"],
            under=err["under"],
            invalid=err["invalid"],
        )
        if isinstance(score, np.ndarray):
            score[np.isnan(score)] = 0
        else:
            if np.isnan(score):
                score = 0
        return score

    def log_loss(self, actual, predicted):
        return np.mean(self.ll(actual, predicted))

    def mean_squared_error(self, actual, predicted):
        sum_square_error = 0.0
        for i in range(len(actual)):
            sum_square_error += (actual[i] - predicted[i]) ** 2.0
        mean_square_error = 1.0 / len(actual) * sum_square_error
        print("MSE: {}".format(mean_square_error))

    def summary(self, input_shape=None):
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
        current_output = input_data
        for idx, layer in enumerate(self.layers):
            ic(idx)
            current_output = layer.run(current_output)

        # set output
        self.output = current_output

    def backward_phase(self, target: np.ndarray):
        # TODO: cek klo misalnya bukan softmax
        current_delta = (self.output - target).reshape((len(target), 1))
        ic(current_delta)

        for idx, layer in enumerate(reversed(self.layers)):
            ic(idx)
            current_delta = layer.compute_delta(current_delta)

        return current_delta

    def update_parameters(self):
        pass
