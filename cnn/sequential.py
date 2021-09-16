from typing import List
from cnn.layer.base import BaseLayer


class Sequential:
    def __init__(self, layers: List[BaseLayer] = []):
        self.type = "Sequential"
        self.layers = layers

    def add(self, layer: BaseLayer):
        self.layers.append(layer)

    def run(self, inputs):
        self.inputs = inputs
        result = inputs
        for layer in self.layers:
            result = layer.run(result)
        return result

    def summary(self, input_shape=None):
        if input_shape is None:
            # TODO: change with commented one when already fixed with multiple data image as inputs
            # n_channel = len(self.inputs[0])
            # length = len(self.inputs[0][0])
            # width = len(self.inputs[0][0][0])
            n_channel = len(self.inputs)
            length = len(self.inputs[0])
            width = len(self.inputs[0][0])
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
