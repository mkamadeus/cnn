from typing import List
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

        # with tqdm(total=len(inputs)) as pbar:
        for i in tqdm(inputs):
            result = i
            for idx, layer in enumerate(self.layers):
                ic(idx, result.shape, result)
                result = layer.run(result)
            final_result.append(result)
            # sleep(0.1)
        #     pbar.update(1)
        # pbar.close()

        return np.array(final_result)

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
