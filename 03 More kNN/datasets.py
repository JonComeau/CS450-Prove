import os.path
import requests
import pandas as pd

datasets = ["cars", "pima", "mpg"]
files = ["cars.data", "pima-indian-diabetes.data", "auto-mpg.data"]

def download():
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"]

    for index in range(len(urls)):
        if not os.path.isfile("data/" + files[index]):
            request = requests.get(urls[index])

            with open('data/' + files[index], 'wb') as f:
                f.write(request.content)


def load(columns):
    data = {}

    for index in range(len(datasets)):
        df = pd.read_csv(
            "data/{}".format(files[index]),
            names=columns[datasets[index]],
            skipinitialspace=True,
            na_values=["?", "NA"],
        )
        data[datasets[index]] = df

    return data
