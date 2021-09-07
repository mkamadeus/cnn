from cnn.layer import ConvolutionalLayer
from cnn.utils import generate_strides
import numpy as np
from icecream import ic

cnn = ConvolutionalLayer(
    np.array(
        [
            [1, 7, 2],
            [11, 1, 23],
            [2, 2, 2],
        ]
    ),
    np.array(
        [
            [1, 1],
            [0, 1],
        ]
    ),
)
ic(cnn)
fm = cnn.convolution(1, 0)
ic(fm)
