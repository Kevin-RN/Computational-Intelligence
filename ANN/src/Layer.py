

class Layer:
    """
    Base class
    """

    def __init__(self):
        self.input = None
        self.output = None

    def feed_forward(self, input):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError