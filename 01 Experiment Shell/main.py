import data
from hard_coded_classifier import HardCodedClassifier

dat = data.load()
x_train, x_test, y_train, y_test = data.split_seventy_thirty(dat.data, dat.target)
model = data.gaussian_nb(x_train, y_train)
prediction = data.predict(model, x_test)
accuracy = data.score(y_test, prediction)

print('Accuracy of GaussianNB:', accuracy)

hard_model = HardCodedClassifier.fit(x_train, y_train)
hard_prediction = HardCodedClassifier.predict(hard_model, x_test)

hard_accuracy = data.score(y_test, hard_prediction)

print('Accuracy of HardCodedClassifier:', hard_accuracy)
