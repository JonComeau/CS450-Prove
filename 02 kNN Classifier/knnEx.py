from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

data = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.targets)
classifier = KNeighborsClassifier(n_neighbors=3)
model = classifier.fit(x_train, y_train)
predictions = model.predict(x_test)

print(predictions)