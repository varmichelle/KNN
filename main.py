from scipy.spatial import distance

class KNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self, X_test, k):
        k_neighbors = self.closest(k)[0:k]
        k_indeces = []
        for i in range(k):
            k_indeces.append(k_neighbors[i][0])
        print k_indeces

    def closest(self, k):
        distances = []
        for i in range(len(self.X_train)):
            distances.append((i,distance.euclidean(i,self.X_train[i])))
        distances = sorted(distances, key=lambda x:x[1])
        return distances

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .75)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNN()

classifier.fit(X_train, Y_train)

k = 5 # change this
predictions = classifier.predict(X_test, k)

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
