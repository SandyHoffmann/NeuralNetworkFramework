"""
    Sandy Hoffmann
    06/03/2024

    Artificial Neural Network (ANN).
    Parameters:
        model (optional): The model to be assigned to the 'layer' attribute.
    Returns:
        None
    Comments:
        * ravel = flatten, leaves the array in a 1D array
    
"""

import numpy as np

class ANN():
    def __init__(self, model = None, range_values = [0, 1]):
        self.layer = model
        self.n_train = int(1e8)
        self.n_eval = int(1e8)
        self.range_values = range_values
    def train(self, training_set):
        for i in range(self.n_train):
            print("train")
            image = next(training_set()).ravel()
            image = self.normalize(image)
            y = self.forward_prop(image)
            print(y)
        pass

    def evaluate(self, evaluation_set):
        for i in range(self.n_eval):
            print("evaluate")
            image = next(evaluation_set()).ravel()
            image = self.normalize(image)
            y = self.forward_prop(image)
            print(y)
        pass

    """
    Perform forward propagation on the input x through the neural network layers and return the output.
    
    Parameters:
    - self: the neural network object
    - x: the input data for forward propagation
    
    Returns:
    - y.ravel(): the output of the forward propagation
    """
    def forward_prop(self, x):
        # Add to two dimensional array
        # before: [1,2,3] => after: [[1,2,3]]
        y = x.ravel()[np.newaxis,:]
        for layer in self.layer:
            y = layer.forward_prop(y)
        return y.ravel()

    """
    Normalize the given value using the range values of the object.
    
    :param value: The value to be normalized
    :return: The normalized value

    Using normalization of values in the range [-0.5, 0.5]
    """

    def normalize(self, value):

        scale = self.range_values[1] - self.range_values[0]
        offset = self.range_values[0]
        return (value - offset) / scale - 0.5
    
    """
    Denormalizes the given value using the range values and returns the denormalized value.
    
    Parameters:
        value: The value to denormalize.
        
    Returns: The denormalized value.
    """

    def denormalize(self, value):
        scale = self.range_values[1] - self.range_values[0]
        offset = self.range_values[0]

        return ((value + 0.5) * scale) + offset