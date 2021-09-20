from typing import List

from numpy.core.fromnumeric import mean
from cnn.layer.base import BaseLayer
from icecream import ic
import numpy as np
from tqdm import tqdm

# tqdm.pandas()


class Sequential:
    def __init__(self, layers: List[BaseLayer] = None):
        self.type = "Sequential"
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def run(self, inputs):

        ic(inputs.shape)

        self.inputs = inputs
        final_result = []

        for i in tqdm(inputs):
            result = i
            for idx, layer in enumerate(self.layers):
                ic(idx, result.shape, result)
                result = layer.run(result)
            final_result.append(result)

        return np.array(final_result)

    def ll(actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        for i in range(0, predicted.shape[0]):
            predicted[i] = min(max(1e-15, predicted[i]), 1 - 1e-15)
        err = np.seterr(all="ignore")
        score = -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        np.seterr(divide=err["divide"], over=err["over"], under=err["under"], invalid=err["invalid"])
        if isinstance(score, np.ndarray):
            score[np.isnan(score)] = 0
        else:
            if np.isnan(score):
                score = 0
        return score

    def log_loss(actual, predicted):
        return np.mean(ll(actual, predicted))

    def mean_squared_error(self, actual, predicted):
        return np.mean(ll(actual, predicted))

    def summary(self, input_shape=None):
        if input_shape is None:
            n_channel = len(self.inputs[0])
            length = len(self.inputs[0][0])
            width = len(self.inputs[0][0][0])
            input_shape = (n_channel, length, width)
        total_weight = 0
        print(f"Model: {self.type}")
        print("----------------------------------------------------")
        print("Layer (type)       Output Shape             Param")
        print("====================================================")
        for index, layer in enumerate(self.layers):
            layer_type = layer.get_type()
            layer_shape, layer_weight = layer.get_shape_and_weight_count(input_shape)
            total_weight += layer_weight
            print(f"{layer_type}        {layer_shape}                 {layer_weight}")
            if index != len(self.layers) - 1:
                print("----------------------------------------------------")
            input_shape = layer_shape
        print("====================================================")
        print(f"Total param/weight: {total_weight}")
