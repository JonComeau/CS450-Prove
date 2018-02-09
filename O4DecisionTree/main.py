import pandas as pd
import json

from sklearn.tree import DecisionTreeClassifier

from O4DecisionTree.DTClassifier import DTClassifier

columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

df = pd.read_csv('../02 kNN Classifier/Data/iris/iris.csv', header=None, names=columns)

x = df.ix[:, 0:4]
y = df['class']

d_tree = DTClassifier(x.iloc[:-1], y.iloc[:-1])

tree_structure = d_tree.structure()
tree_json_str = json.dumps(tree_structure, indent=2, sort_keys=True)
# print(tree_json_str)

predicted_class = d_tree.predict(x.iloc[-1:])

sk_tree = DecisionTreeClassifier()
sk_tree = sk_tree.fit(x.iloc[:-1], y.iloc[:-1])
sk_predicted_class = sk_tree.predict(x.iloc[-1:])[0]

print(f"\nsklearn Class: {sk_predicted_class}, My Class: {predicted_class}, Original Class: {list(y)[-1:][0]}")
