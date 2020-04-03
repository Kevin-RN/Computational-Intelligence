import numpy as np
from src.Layer import Layer


class ActivationLayer(Layer):
    """
    Can choose either from sigmoid or tanh
    Inherit from base class Layer
    """

    def __init__(self, activation, derivative_activation):
        self.activation = activation
        self.derivative_activation = derivative_activation

    def feed_forward(self, input_data):
        """
        Feeds forward the input data
        Computes the output Y of a layer for a given input X
        :param input_data: input data
        :return: activated input
        """
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        Backward propagate
        Computes input error for a given output error (and update parameters if any) which is why it's called back prop.
        :param output_error: dE/dY
        :param learning_rate: Not used
        :return: input_error=dE/dX for a given output_error=dE/dY.
        """
        return self.derivative_activation(self.input) * output_error


# Activation functions and its derivative functions
def tanh(x):
    return np.tanh(x)


def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def sigmoid_func(z):
    return 1.0 / (1.0 + np.exp(-z))


def derivative_sigmoid(z):
    sigmoid = sigmoid_func(z)
    return sigmoid * (1 - sigmoid)

