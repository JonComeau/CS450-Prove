import operator
import numpy as np


def distance(x, y):
    dist = 0
    for index in range(len(y) - 1):
        dist += pow(x[index] - y[index], 2)
    return np.sqrt(dist)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


class kRegression:
    def fit(self, train_data, train_classes):
        self.data = train_data[:, :-1] = train_classes

    def predict(self, input):
        distances = []

        for index in range(len(self.data)):
            distances.append((distance(input, self.data[index]), index))

        distances.sort(key=operator.itemgetter(0))

        neighbors = []

        for index in range(9):
            neighbors.append(self.data[distances[index][1]])

        return sum(neighbors) / len(neighbors)
