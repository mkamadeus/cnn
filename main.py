import cnn
from cnn.layer import ConvolutionalLayer
# from cnn.layer import PoolingLayer
import numpy as np
from icecream import ic

model = cnn.Sequential()
model.add(
    ConvolutionalLayer(
        input_shape=(3, 3, 3),
        padding=0,
        filter_count=2,
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
# TODO: this is a single instance input. How about multiple instances?
result = model.run(
    inputs=np.array(
        [
            [
                [1, 7, 2],
                [11, 1, 23],
                [2, 2, 2],
            ],
            [
                [1, 7, 2],
                [11, 1, 23],
                [2, 2, 2],
            ],
            [
                [1, 7, 2],
                [11, 1, 23],
                [2, 2, 2],
            ],
        ]
    )
)
ic(result)
ic(result.shape)
