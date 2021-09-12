from cnn import activations
import cnn
from cnn.layer import ConvolutionalLayer
# from cnn.layer import PoolingLayer
import numpy as np
from icecream import ic

model = cnn.Sequential()
model.add(
    ConvolutionalLayer(
        input_shape=(3, 3),
        activation="relu",
        padding=0,
        filter_count=3,
        kernel_shape=(2, 2),
        stride=1,
    )
)
# model.add(
#     PoolingLayer(
#         size=(2, 2),
#         stride=1,
#         mode="max",
#     )
# )
result = model.run(
    inputs=np.array(
        [
            [1, 7, 2],
            [11, 1, 23],
            [2, 2, 2],
        ]
    )
)
ic(result)
