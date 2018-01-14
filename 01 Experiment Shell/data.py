from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load():
    iris = datasets.load_iris()
    return iris


def split_seventy_thirty(data, labels):
    return train_test_split(data, labels)


def gaussian_nb(x_train, y_train):
    classifier = GaussianNB()

    return classifier.fit(x_train, y_train)


def predict(model, x_test):
    return model.predict(x_test)


def score(test, pred):
    return accuracy_score(test, pred)
