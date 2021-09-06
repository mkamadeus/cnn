from cnn.utils import generate_strides
import numpy as np
from icecream import ic
from numpy.lib.stride_tricks import as_strided

mat = np.array(
    [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5],
    ]
)

ic(generate_strides(mat, (2, 2)))
