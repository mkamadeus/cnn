import numpy as np


class BaseLayer:
    """
    Defines base class for a layer.
    """

    def __init__(self):
        return

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return inputs
