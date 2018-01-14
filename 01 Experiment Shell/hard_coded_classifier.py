from random import Random, randint

from sklearn.naive_bayes import GaussianNB


class HardCodedClassifier():
    @staticmethod
    def fit(x_train, y_train):
        return GaussianNB()

    @staticmethod
    def predict(model, x_test):
        return [randint(0, 1) for i in x_test]
