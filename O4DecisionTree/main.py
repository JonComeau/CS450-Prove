import pandas as pd
import json

from O4DecisionTree import DecisionTreeClassifier

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv('../02 kNN Classifier/Data/iris/iris.csv', header=None, names=columns)

x = df.ix[:, 0:4]
y = df['class']

tree = DecisionTreeClassifier.DecisionTreeClassifier(x.iloc[:-1], y.iloc[:-1])

print("\n\n\nStart Print")

tree_structure = tree.structure()
tree_json_str = json.dumps(tree_structure, indent=2, sort_keys=True)
# print(tree_json_str)

predicted_class = tree.predict(x.iloc[-1:])

print("Predicted Class: {}, Real Class: {}".format(predicted_class, list(y)[-1:][0]))
