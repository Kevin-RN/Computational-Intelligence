import numpy as np
from scipy import linalg


# Perceptron class
class Perceptron:

    # Constructor for Perceptron class
    # @param dimensions Size of the input values to classify
    # @param threshold The number of epochs to iterate (standard 50 epochs)
    # @param learning_rate The magnitude in which the weights will change (standard 0.01)
    def __init__(self, dimensions, threshold=50, learning_rate=0.01):
        self.weights = np.zeros(dimensions)
        self.bias = 0
        self.threshold = threshold
        self.learning_rate = learning_rate

    # Predicts the expected class of the input
    # @inputs Input values
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        if summation > 0:
            return 1
        return 0

    # Trains the Perceptron object
    # @param labels Expected output
    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.bias += self.learning_rate * (label - prediction)
                self.weights += self.learning_rate * (label - prediction) * inputs


if __name__ == "__main__":
    # 2.1
    inputs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
    )
    # labels: AND, OR, XOR
    labels = np.array([
        [0, 0, 0, 1],
        [0, 1, 1, 1],
        [0, 1, 1, 0]
    ])
    # Create perceptron with 2 dimensions
    perceptron = Perceptron(2)
    # Train perceptron with all possible input values and expected outcome
    perceptron.train(inputs, labels[0])
    # print out prediction
    print("predicted: " + str(perceptron.predict(inputs[0])) + ", actual value: " + str(labels[0][0]))

    exit()
