from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
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

# cars

cars_class_column = "class-values"

process_cars(datas['cars'])

cars_data, cars_class = split(datas['cars'], cars_class_column)

cars_data_train, cars_data_test, cars_class_train, cars_class_test = train_test_split(cars_data, cars_class)

print("Cars dataset with sklearn kNN classifier started")

cars_sklearn_kNN = KNeighborsClassifier(n_neighbors=5)
cars_sklearn_kNN.fit(cars_data_train, cars_class_train.ravel())
cars_sklearn_score = cars_sklearn_kNN.score(cars_data_test, cars_class_test)

print("Cars dataset with my kNN classifier started")

cars_kNN = KNNClassifier(5)
cars_kNN.fit(cars_data_train, cars_class_train.ravel())
cars_score = cars_kNN.score(cars_data_test, cars_class_test)

print(f"sklearn kNN accuracy: {cars_sklearn_score}, my kNN accuracy: {cars_score}\n")

# pima

pima_class_column = "class-variable"

process_pima(datas['pima'], ['glucose-concentration',
                             'diastolic-blood-pressure',
                             'triceps-skin-thickness',
                             'bmi',
                             'diabetes-pedigree-function',
                             'two-hour-insulin',
                             'age'])

# mpg
