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
    if size < 0:
        raise Exception("size should be bigger than 0")
    if size == 0:
        return mat

    padded = np.pad(mat, size, pad_with, padder=padder)
    return padded


def generate_strides(mat: np.array, kernel_size: Tuple[int, int], stride: int = 1):
    """
    Generates possible array strides from given kernel size
    """
    # calculates view shape and strides for as_strided
    view_shape = tuple(np.subtract(mat.shape, kernel_size) + 1) + kernel_size
    view_strides = mat.strides + mat.strides

    # generate all view, strides every column and row
    result = as_strided(mat, strides=view_strides, shape=view_shape)[::stride, ::stride]

    return result


def generate_random_uniform_matrixes(n_matrix: int, size: Tuple[int, int]):
    """
    Generates n random uniform matrixes from given kernel size
    """
    return np.array([np.random.uniform(low=-1.0, high=1.0, size=size) for i in range(n_matrix)])
