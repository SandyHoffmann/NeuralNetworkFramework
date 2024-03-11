import numpy as np

class Tahn():
    def __init__(self):
        pass
    """
    Calculate the hyperbolic tangent of the input array.
    Parameters:
        self: the instance of the class
        inputs: the input array (in numpy form)
    Returns:
        The hyperbolic tangent of the input array.
    """
    def calc(self, inputs):
        return np.tanh(inputs)