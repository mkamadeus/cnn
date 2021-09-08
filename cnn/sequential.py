from typing import List

import numpy as np
from cnn.layer.base import BaseLayer


class Sequential:
    def __init__(self, layers: List[BaseLayer] = []):
        self.layers = layers

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def run(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.run(result)
        return result

    def summary(self):
        print(f"model {self.layers}")
        print(self.layers)
