from typing import Tuple
import numpy as np
from numpy.lib.stride_tricks import as_strided


def pad_with(vector, pad_width, _, kwargs):
    """
    Helper function for adding padding to a Numpy array.
    """
    pad_value = kwargs.get("padder", 10)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value


def pad_array(mat: np.array, size: int, padder: int):
    """
    Pads a 2D array with `padder` as the number pads.
    """
    padded = np.pad(mat, size, pad_with, padder=padder)
    return padded


# TODO: fix stride, should stride vertically as well
def generate_strides(mat: np.array, kernel_size: Tuple[int, int], stride: int = 1):
    """
    Generates possible array strides from given kernel size
    """
    view_shape = tuple(np.subtract(mat.shape, kernel_size) + 1) + kernel_size
    view_strides = mat.strides + mat.strides

    strided_matrices = as_strided(mat, strides=view_strides, shape=view_shape)
    strided_shape = strided_matrices.shape
    result = strided_matrices.reshape((strided_shape[0] * strided_shape[1], strided_shape[2], strided_shape[3]))

    return result
