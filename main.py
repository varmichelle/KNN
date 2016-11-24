import random

class KNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.Y_train)
            predictions.append(label)
        return predictions

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .75)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNN()

classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
