from datasets import download, load
from kNN import KNNClassifier

import pandas as pd
import numpy as np

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

carDF = datas['cars']

carDF['buying'] = carDF['buying'].astype('category')
carDF['maint'] = carDF['maint'].astype('category')
carDF['lug-boot'] = carDF['lug-boot'].astype('category')
carDF['safety'] = carDF['safety'].astype('category')

carDF["buying_cat"] = carDF["buying"].astype('category').cat.codes
carDF["maint_cat"] = carDF["maint"].astype('category').cat.codes
carDF["lug-boot_cat"] = carDF["lug-boot"].astype('category').cat.codes
carDF["safety_cat"] = carDF["safety"].astype('category').cat.codes

# pima

pimaDF = datas['pima']

pimaDF['glucose-concentration'].replace("0", "NaN")
pimaDF['diastolic-blood-pressure'].replace("0", "NaN")
pimaDF['triceps-skin-thickness'].replace("0", "NaN")
pimaDF['bmi'].replace("0", "NaN")
pimaDF['diabetes-pedigree-function'].replace("0", "NaN")
pimaDF['two-hour-insulin'].replace("0", "NaN")
pimaDF['age'].replace("0", "NaN")

# mpg

mpgDF = datas['mpg']

mpgDF2 = mpgDF

# k-fold cross-validation

dataset = input("Which dataset do you want to use: (C)ars, (P)ima, (M)pg: ")
k = input("What is the value for k: ")

if k.lower() == "c":
    data = datas['cars']
elif k.lower() == "p":
    data = datas['pima']
else:
    data = datas['mpg']

