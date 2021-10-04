import numpy as np


def mean_squared_error(output: np.ndarray, target: np.ndarray):
    """
    Calculate MSE, using the variant of 0.5 as a factor.
    """
    return 0.5 * np.sum((output - target) ** 2)
