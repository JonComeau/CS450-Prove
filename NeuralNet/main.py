from sklearn import datasets, preprocessing

import pandas as pd
from NeuralNet.Net import NNetClassifier

# Grabbing iris dataset
iris = datasets.load_iris()

# normalizing data --Retrieved from https://stackoverflow.com/a/37199623/7747350--
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(iris["data"])
normalized = pd.DataFrame(np_scaled)

# using Neural net to classify
net = NNetClassifier([2, 3, 1])

print(net.calculate(iris["data"][0]))

print(net.calculate_all(iris["data"]))
