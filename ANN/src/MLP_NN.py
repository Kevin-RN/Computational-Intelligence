from src.Layer import Layer
from src.MLP_data import MLPData
from src.ActivationLayer import ActivationLayer, tanh, derivative_tanh
import numpy as np


class FCLayer(Layer):
    """
    Inherit from base class Layer.
    """

    def __init__(self, input_size, output_size):
        """
        Initializes the weights matrix with random values between -0.5 and 0.5 and bias vector with zeros accordingly
         to the input size and output size.
        :param input_size: number of input neurons
        :param output_size: number of output neurons
        """
        # set up weight matrix with random numbers between -0.5 and 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5
        # Different weight initializations
        # self.weights = np.random.randint(5, size=(input_size, output_size)) - 2.5
        # self.weights = np.random.randint(7, size=(input_size, output_size)) - 3.5
        # self.weights = np.random.randint(8, size=(input_size, output_size)) - 4
        # set up bias vector with zeroes
        self.bias = np.zeros((1, output_size))

    def feed_forward(self, input_data):
        """
        x*W + b
        :param input_data: the feature values
        :return: a
        """
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """
        Computes dE/dW, dE/dB given a output_error
        :param output_error: dE/dY
        :param learning_rate: 0.01
        :return: input_error=dE/dX.
        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update weights and bias
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class NeuralNetwork:
    def __init__(self):
        """
        Set up the layer input and output size
        """
        self.layers = []

    def add(self, layer):
        """
        Adds a single layer to the network
        :param layer: a pair of input size and output size
        :return: nothing but initializes new random weights and biases
        """
        self.layers.append(layer)

    def cost(self, expected, predicted):
        """
        Loss function
        :param expected: expected output
        :param predicted: predicted output
        :return: cost
        """
        # return np.mean(np.power(y_true-y_pred, 2))
        return np.linalg.norm(expected - predicted)

    def derivative_cost(self, expected, predicted):
        """
        Derivative of loss function
        :param expected: expected output
        :param predicted: predicted output
        :return: cost derivative
        """
        # return 2 * (predicted - expected) / expected.size
        return predicted - expected

    def predict(self, input_data):
        """
        Predict output for given input
        :param input_data: the train data of features
        :return: prediction matrix (vector of each prediction)
        """
        # size of input data
        size = len(input_data)
        result = []

        # run network over all samples
        for i in range(size):
            # feed_forward
            output = input_data[i]
            for layer in self.layers:
                output = layer.feed_forward(output)
            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate):
        """
        Train the neural network and update the weights and bias for prediction later
        :param x_train: feature vectors
        :param y_train: label vectors
        :param epochs: repeat
        :param learning_rate: 0.01
        :return:
        """
        # size of examples
        size = len(x_train)

        err_pre = 0

        # training loop
        for i in range(epochs):
            err = 0
            for j in range(size):
                # feed_forward
                output = x_train[j]
                for layer in self.layers:
                    output = layer.feed_forward(output)

                # cost (to check error)
                err += self.cost(y_train[j], output)

                # backward propagation
                # updates weights and bias for prediction later
                error = self.derivative_cost(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate error
            err /= size

            print('epoch %d error=%f' % (i+1, err))
            # print('epoch %d: accuracy=%f' % (i+1, 1-err))

            # Stop earlier if error change is insignificant
            if abs(err_pre - err) < 0.0003:
                print('End at epoch %d with error=%f' % (i+1, err))
                break
            err_pre = err

    def get_product_num(self, predicted_output):
        """
        Convert predict vector to predicted output
        :param predicted_output: vector of numbers
        :return: a number between 1-7
        """
        return np.argmax(predicted_output) + 1

    def prediction_vector_to_num_list(self, mat):
        """
        Change a list of prediction vector to a list of labels
        :param mat: matrix of vectors of predictions
        :return: a list of labels
        """
        predicted_label = list()
        for pred_vec in np.asarray(mat):
            predicted_label.append(self.get_product_num(pred_vec))
        return predicted_label

    def comp_arrays(self, a, b):
        """
        Get accuracy
        :param a: list 1
        :param b: list 2
        :return: accuracy percentage
        """
        arr = np.equal(a, b)
        trues = np.sum(arr)
        return trues / len(arr)


def main():
    # 2.2
    # Get data
    data_object = MLPData()
    # Split to train, validation, and test data
    data_object.split_data()
    x_train, y_train, x_test, y_test = \
        data_object.get_x_train(), data_object.get_y_train(), data_object.get_x_test(), data_object.get_y_test()

    # (5497, 10) (5497, 7) (1178, 10) (1178, 7)
    print('x_train, y_train, x_test, y_test:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # reshape and normalize input data to (num_sample, 1, num_input_neurons)
    x_train = x_train.reshape(x_train.shape[0], 1, 10)
    x_train = x_train.astype('float32')
    x_train /= 10

    x_test = x_test.reshape(x_test.shape[0], 1, 10)
    x_test = x_test.astype('float32')
    x_test /= 10

    np.random.seed(99)

    # Network
    # use tanh as activation function
    nn = NeuralNetwork()
    nn.add(FCLayer(10, 7))
    nn.add(ActivationLayer(tanh, derivative_tanh))

    nn.add(FCLayer(7, 7))
    nn.add(ActivationLayer(tanh, derivative_tanh))

    # Train
    nn.train(x_train, y_train, epochs=35, learning_rate=0.01)

    # Predict on y_test
    out = nn.prediction_vector_to_num_list(nn.predict(x_test))

    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(nn.prediction_vector_to_num_list(y_test))

    print("Final accuracy: ", nn.comp_arrays(out, nn.prediction_vector_to_num_list(y_test)))

    # Predict on unknown set
    # out = nn.prediction_vector_to_num_list(nn.predict(data_object.load_unknown_dataset()))
    # np.savetxt("Group_27_classes.txt", [out], delimiter=',', fmt='%i')


if __name__ == '__main__':
    main()
    exit()
