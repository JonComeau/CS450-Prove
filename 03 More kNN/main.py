from sklearn.preprocessing import LabelEncoder
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

cars_class_column = "class_values"

process_cars(datas['cars'])

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



# k-fold cross-validation

# cars
