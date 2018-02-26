from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from datasets import download, load
from kNN import KNNClassifier

import pandas as pd
import numpy as np
import decimal


def process_cars(df):
    enc = LabelEncoder()
    columns = df.columns.values

    for index in range(len(columns) - 1):
        new_col = enc.fit_transform(df[columns[index]])
        df[columns[index]] = new_col


def split(df, class_column):
    data = df.as_matrix([column for column in df.columns.values if column != class_column])
    classes = df.as_matrix([class_column])

    return data, classes


def process_pima(pimaDF, columns):
    for column in columns:
        mean = 0
        value_count = 0
        precision = 0
        for row in pimaDF[column]:
            if row != 0:
                row_precision = str(row)[::-1].find('.')

                if row_precision != -1 and row_precision > precision:
                    precision = row_precision

                mean += row
                value_count += 1

        mean /= value_count
        mean = round(mean, precision)

        for index in range(len(pimaDF[column])):
            if pimaDF[column][index] == 0:
                pimaDF.at[index, column] = mean


def calculate_accuracy(df, class_column, neighbors=5):
    data, classes = split(df, class_column)

    data_train, data_test, class_train, class_test = train_test_split(data, classes)

    sklearn_kNN = KNeighborsClassifier(n_neighbors=neighbors)
    sklearn_kNN.fit(data_train, class_train.ravel())
    sklearn_score = sklearn_kNN.score(data_test, class_test)

    kNN = KNNClassifier(neighbors)
    kNN.fit(data_train, class_train.ravel())
    score = kNN.score(data_test, class_test)

    print(f"sklearn kNN accuracy: {sklearn_score}, my kNN accuracy: {score}\n")


cols = {
    "cars": [
        "buying",
        "maint",
        "doors",
        "persons",
        "lug-boot",
        "safety",
        "class-values"],
    "pima": [
        "times-pregnant",
        "glucose-concentration",
        "diastolic-blood-pressure",
        "triceps-skin-thickness",
        "two-hour-insulin",
        "bmi",
        "diabetes-pedigree-function",
        "age",
        "class-variable"
    ],
    "mpg": [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model-year",
        "origin",
        "car-name"
    ]
}

# Loading Data into program

download()

datas = load(cols)
iris = load_iris()

# testing iris

irisDF = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

print("Iris Dataset")

calculate_accuracy(irisDF, "target")

# cars

process_cars(datas['cars'])

print("Cars Dataset")

calculate_accuracy(datas['cars'], "class-values")

# pima indian

process_pima(datas['pima'], ['glucose-concentration',
                             'diastolic-blood-pressure',
                             'triceps-skin-thickness',
                             'bmi',
                             'diabetes-pedigree-function',
                             'two-hour-insulin',
                             'age'])

print("Pima Dataset")

calculate_accuracy(datas['pima'], "class-variable")

# mpg


