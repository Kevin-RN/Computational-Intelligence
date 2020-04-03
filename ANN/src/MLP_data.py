import numpy as np
# from sklearn.model_selection import train_test_split


class MLPData:
    # Constructor for Multilayer Perceptron class
    def __init__(self):
        self.features = np.genfromtxt("../data/features.txt", delimiter=",")
        self.targets = np.genfromtxt("../data/targets.txt", delimiter=",")
        self.labels = np.unique(self.targets)
        self.test_fraction = 0
        self.validation_fraction = 0
        self.training_fraction = 0
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = \
            np.zeros(10), np.zeros(10), np.zeros(10), np.zeros(1), np.zeros(1), np.zeros(1)

    # Splitting the features and targets into training, validation, and test set
    def split_data(self, test_fraction=0.15, validation_fraction=0.15, training_fraction=0.7):
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.training_fraction = 1 - test_fraction - validation_fraction  # 0.7

        # checking fraction
        assert 0.0 < test_fraction < 1.0, f"The test_fraction must be in the range (0.0, 1.0) but is {test_fraction}"
        assert 0.0 < validation_fraction < 1.0, f"The validation_fraction must be in the range (0.0, 1.0) but is " \
                                                f"{validation_fraction}"
        assert 0.0 < training_fraction < 1.0, f"The training_fraction must be in the range (0.0, 1.0) but is " \
                                              f"{training_fraction}"

        # append target accordingly to each sample example, before shuffling
        data_set = np.column_stack([self.features, self.targets])
        np.random.shuffle(data_set)

        # split the data set with the ratio above
        test_size = test_fraction * len(data_set)
        validation_size = validation_fraction * len(data_set)
        training_idx = np.random.randint(data_set.shape[0], size=int(len(data_set)-test_size-validation_size))
        validation_idx = np.random.randint(data_set.shape[0], size=int(validation_size))
        test_idx = np.random.randint(data_set.shape[0], size=int(test_size))
        training, validation, test = data_set[training_idx, :], data_set[validation_idx, :], data_set[test_idx, :]
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = \
            training[:, :10], validation[:, :10],  test[:, :10], \
            training[:, 10:].flatten(), validation[:, 10:].flatten(), test[:, 10:].flatten()

    def split_cross_validation(self, k=7):
        # append target accordingly to each sample example, before shuffling
        data_set = np.column_stack([self.features, self.targets])
        np.random.shuffle(data_set)
        return np.split(data_set, k)

    def load_unknown_dataset(self):
        return np.genfromtxt("../data/unknown.txt", delimiter=",")

    def get_x_train(self):
        return self.x_train

    def get_x_validation(self):
        return self.x_val

    def get_x_test(self):
        return self.x_test

    def label_to_list(self, l):
        result = list()
        for idx, val in enumerate(l):
            single = np.zeros(7)
            single[int(val)-1] = 1
            result.append(single)
        return np.asarray(result)

    def get_y_train(self):
        return self.label_to_list(self.y_train)

    def get_y_validation(self):
        return self.label_to_list(self.y_val)

    def get_y_test(self):
        return self.label_to_list(self.y_test)

    def compute_confusion_matrix(self, actual, predicted):

        # calculate the confusion matrix
        cm = np.zeros((8, 8))
        for a, p in zip(actual, predicted):
            cm[a][p] += 1

        return cm
