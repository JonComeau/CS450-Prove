from random import randint
from collections import Counter

import numpy as np
import math
import operator


def distance(x, y):
    dist = 0
    for index in range(len(y)):
        dist += pow(x[index] - y[index], 2)
    return math.sqrt(dist)


class KNNClassifier:
    def __init__(self, n_neighbors):
        self.k = n_neighbors
        self.max_min = []

    def fit(self, train_data, train_target):
        self.train_data = train_data.astype(dtype=np.float)
        self.train_target = train_target

        columns = self.train_data.T

        for col in columns:
            self.get_max_min(col)

        print(self.max_min)

        for index in range(len(columns)):
            for ind in range(len(columns[index])):
                value = float(columns[index][ind])
                col_min = float(self.max_min[index]["min"])
                col_max = float(self.max_min[index]["max"])
                total = (value - col_min) / (col_max - col_min)

                columns[index][ind] = total

        print(self.train_data)

    def get_max_min(self, col):
        col_min = min(col)
        col_max = max(col)

        self.max_min.append({"max": col_max, "min": col_min})

    def predict(self, test_data):
        predictions = []

        for test_row in test_data:
            distances = []

            for index in range(len(self.train_data)):
                distances.append((distance(test_row, self.train_data[index]), index))

            distances.sort(key=operator.itemgetter(0))

            neighbors = []

            for index in range(self.k):
                neighbors.append(self.train_target[distances[index][1]])

            counter = Counter(neighbors)
            most_common = counter.most_common(1)

            print(neighbors)

            predictions.append(most_common[0][0])

        return predictions

    def score(self, test_data, test_classes):
        predicted = self.predict(test_data)

        correct_count = 0

        for index in range(len(test_data)):
            if test_classes[index] == predicted[index]:
                correct_count += 1

        return correct_count / len(test_classes)
