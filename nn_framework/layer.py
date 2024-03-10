"""
    Sandy Hoffmann
    08/03/2024

    Dense Layer.
    Parameters:
        n_input (optional): The number of inputs to the layer.
        n_output (optional): The number of outputs of the layer.
    Returns:
        None
    Comments:
        * layer that takes in n_input and outputs n_output
        * The inputs receives a +1 value because the bias is added
"""
import numpy as np

class Dense():
    def __init__(self, n_input=1, n_output=1):
        self.n_input = n_input
        self.n_output = n_output
        self.weight_scale = 1
        self.weights = self.weight_scale * np.random.sample(
            size=(self.n_input+1, self.n_output)
        )*2 - 1
        self.x = np.zeros((1, self.n_input+1))
        self.y = np.zeros((1, self.n_output))