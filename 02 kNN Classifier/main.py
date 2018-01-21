import numpy as np
import pandas as pd
import sys
import math

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from random import randint


class KNNClassifier:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target

    def predict(self, test_data):
        classes = []

        for item in test_data:
            dists = []
            dist_dict = {}

            for point in self.train_data:
                point_dists = []

                for index in range(len(item)):
                    point_dists.append(np.linalg.norm(item[index] - point[index]))

                dists.append(sum(point_dists))

            for index in range(len(dists)):
                # print(str(dists[index]) + ', ')
                dist_dict[dists[index]] = index

            dists.sort()

            neigh_classes = []

            for index in range(self.k):
                ind = dist_dict[dists[index]]

                neigh_classes.append(self.train_target[ind])

            cls_count = dict((cls, 0) for cls in set(self.train_target))

            for cls in neigh_classes:
                cls_count[cls] += 1

            largest_cls = []
            counter = 0

            for key, value in cls_count.items():
                if value > counter:
                    counter = value
                    largest_cls.append(key)

            if len(largest_cls) > 1:
                rand = randint(0, 1)
                largest = largest_cls[rand]
            else:
                largest = largest_cls[0]

            classes.append(largest)

        return classes


class KDNode:
    def __init__(self, point, dim, par):
        self.point = point
        self.left = None
        self.right = None
        self.dim = dim
        self.par = par


class KDTree:
    def __init__(self, points, k):
        self.points = points
        self.dim = k
        self.current_best_dist = sys.float_info.max
        self.current_best = None

    def construct(self, points, depth, parent):
        dim = depth % (self.dim - 1)

        if len(points) == 0:
            return None
        if len(points) == 1:
            return KDNode(points[0], dim, parent)

        np.sort(points, axis=dim)

        median = int(len(points) // 2)

        node = KDNode(points[median], dim, parent)
        node.left = self.construct(points[0:median], depth + 1, node)
        node.right = self.construct(points[median:], depth + 1, node)

        return node

    def nearest_neighbor(self, root, point, depth):
        if root.left is None:
            dist = distance(root.point, point)
            return root.point, dist, 0
        else:
            axis = depth % (self.dim - 1)

            if float(point[axis]) < float(root.point[axis]):
                guess, dist, height = self.nearest_neighbor(root.left, point, depth + 1)
            else:
                guess, dist, height = self.nearest_neighbor(root.right, point, depth + 1)

        if height < 3:
            if float(point[axis]) < float(root.point[axis]):
                guess2, dist2, height2 = self.nearest_neighbor(root.right, point, depth + 1)
            else:
                guess2, dist2, height2 = self.nearest_neighbor(root.left, point, depth + 1)
        else:
            dist2 = sys.float_info.max

        dist3 = distance(root.point, point)

        if dist3 < dist2:
            dist2 = dist3
            guess2 = root.point

        if dist2 < dist:
            dist = dist2
            guess = guess2

        return guess, dist, height + 1


def distance(point1, point2):
    dist = 0

    for indx in range(len(point1) - 1):
        dist += (float(point1[indx]) ** 2) - (float(point2[indx]) ** 2)

    return dist


columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv('Data/iris/iris.csv', header=None, names=columns)

x = np.array(df.ix[:, 0:4])
y = np.array(df['class'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

tree_path = input("Would you like to use a kdtree to find the nearest neighbors? (Y or N): ")

# test input
# tree_path = 'y'

if tree_path.lower() == 'y':
    data = [[] for item in x_train]

    for index in range(len(x_train)):
        for item in x_train[index]:
            data[index].append(item)
        data[index].append(y_train[index])

    data = np.array(data)

    tree = KDTree(data, 3)
    tree_root = tree.construct(tree.points, 0, None)
    best_guess, best_dist, best_height = tree.nearest_neighbor(tree_root, x_train[0], 0)

    print(best_guess)
else:
    # Testing normal output
    classifier = KNeighborsClassifier(n_neighbors=3)
    model = classifier.fit(x_train, y_train)
    predictions = model.predict(x_test)

    print(predictions)

    clss = KNNClassifier(3)
    clss.fit(x_train, y_train)
    predict = clss.predict(x_test)

    print(predict)
