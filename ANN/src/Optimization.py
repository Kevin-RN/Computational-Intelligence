from src.Layer import Layer
from src.MLP_data import MLPData
from src.MLP_NN import NeuralNetwork
from src.MLP_NN import FCLayer
from src.ActivationLayer import ActivationLayer, tanh, derivative_tanh, sigmoid_func, derivative_sigmoid
import numpy as np
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool
from multiprocessing.dummy import Pool as Poolf


class CrossValidation:
    def __init__(self, k=7):
        self.k = k
        self.mlp_data = MLPData()
        self.data_split = np.array(self.mlp_data.split_cross_validation(k))

    def get_x_train(self, j):
        ret = np.delete(self.data_split, j, 0)
        ret = ret.reshape(ret.shape[0]*ret.shape[1], ret.shape[2])
        ret = ret[:, :10]
        ret = ret.reshape(ret.shape[0], 1, 10)
        ret = ret.astype('float32')
        ret /= 10
        return ret

    def get_y_train(self, j):
        ret = np.delete(self.data_split, j, 0)
        ret = ret.reshape(ret.shape[0] * ret.shape[1], ret.shape[2])
        ret = ret[:, 10:].flatten()
        return self.label_to_list(ret)

    def get_x_test(self, j):
        ret = np.copy(self.data_split[j])
        ret = ret[:, :10]
        ret = ret.reshape(ret.shape[0], 1, 10)
        ret = ret.astype('float32')
        ret /= 10
        return ret

    def get_y_test(self, j):
        ret = np.copy(self.data_split[j])
        ret = ret[:, 10:].flatten()
        return self.label_to_list(ret)

    def get_k(self):
        return self.k

    def label_to_list(self, label):
        return self.mlp_data.label_to_list(label)


def run_nn(cross_validation, j, i):
    print("Neurons: ", i, ", fold no. ", j)

    x_train, y_train, x_test, y_test = \
        cross_validation.get_x_train(j), cross_validation.get_y_train(j), \
        cross_validation.get_x_test(j), cross_validation.get_y_test(j)

    # Init neural network
    nn = NeuralNetwork()
    nn.add(FCLayer(10, i))
    nn.add(ActivationLayer(tanh, derivative_tanh))

    nn.add(FCLayer(i, 7))
    nn.add(ActivationLayer(tanh, derivative_tanh))

    # Train
    nn.train(x_train, y_train, epochs=35, learning_rate=0.01)

    # Predict
    out = nn.prediction_vector_to_num_list(nn.predict(x_test))
    acc = nn.comp_arrays(out, nn.prediction_vector_to_num_list(y_test))

    # print(" predicted values : ")
    # print(out, end="\n")
    # print("true values : ")
    # print(nn.prediction_vector_to_num_list(y_test))
    print("  accuracy: ", acc, '\n')
    return acc


def run_k_fold(cross_validation, j, i):
    accuracy_k = 0
    repetitions = 10

    # Run each repetition of training a nn on a separate process (big speedup!!)
    with Pool(repetitions) as p:
        accuracy_k = sum(p.starmap(run_nn, [(cross_validation, j, i) for _ in range(repetitions)]))

    accuracy_k /= repetitions
    print("accuracy of k-fold: ", accuracy_k)
    return accuracy_k


def main():
    cross_validation = CrossValidation()
    k = cross_validation.get_k()

    min_neurons = 7
    max_neurons = 31

    accuracy_neurons = np.zeros(31)
    # start_time = time.time()
    for i in range(min_neurons, max_neurons, 7):
        accuracy_n = 0

        # Spawn threads to calculate the k-fold validation for current number of neurons
        with Poolf(k) as pool:
            accuracy_n = sum(pool.starmap(run_k_fold, [(cross_validation, j, i) for j in range(k)]))

        # Average out accuracy over k
        accuracy_n /= k
        accuracy_neurons[i] = accuracy_n

    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(accuracy_neurons)

    x_coordinate = np.nonzero(accuracy_neurons)[0]
    y_coordinate = accuracy_neurons[x_coordinate]

    # print(x_coordinate)
    # print(y_coordinate)

    plt.plot(x_coordinate, y_coordinate, linestyle='-', marker='o')
    plt.xticks(np.arange(1, 31, step=2))
    plt.ylim((np.amin(y_coordinate)-0.01, np.amax(y_coordinate)+0.01))
    plt.xlim(min_neurons-1, max_neurons)
    plt.xlabel('No. of Neurons')
    plt.ylabel('Avg. Accuracy')
    plt.savefig('avg_accuracy.png')
    # plt.show()


if __name__ == '__main__':
    main()
    exit()
