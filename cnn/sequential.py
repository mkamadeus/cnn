from typing import List
from cnn.layer.base import BaseLayer
from icecream import ic
import numpy as np


class Sequential:
    def __init__(self, layers: List[BaseLayer] = []):
        self.layers = layers

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def run(self, inputs):

        ic(inputs.shape)

        final_result = []

        for i in inputs:
            result = i
            for idx, layer in enumerate(self.layers):
                ic(idx, result.shape, result)
                result = layer.run(result)
            final_result.append(result)

        return np.array(final_result)

    def summary(self):
        print(f"model {self.layers}")
        print(self.layers)
