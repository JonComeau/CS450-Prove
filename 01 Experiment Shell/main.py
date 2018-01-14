import data
from hard_coded_classifier import HardCodedClassifier


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


dat = data.load()
x_train, x_test, y_train, y_test = data.split_seventy_thirty(dat.data, dat.target)

accuracies = []

for i in range(100):
    model = data.gaussian_nb(x_train, y_train)
    prediction = data.predict(model, x_test)
    accuracies.append(data.score(y_test, prediction))

print('Accuracy of GaussianNB:', mean(accuracies))

hard_model = HardCodedClassifier.fit(x_train, y_train)
hard_prediction = HardCodedClassifier.predict(hard_model, x_test)

hard_accuracy = data.score(y_test, hard_prediction)

print('Accuracy of HardCodedClassifier:', hard_accuracy)
