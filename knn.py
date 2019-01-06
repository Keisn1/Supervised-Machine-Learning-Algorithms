import numpy as np
from sortedcontainers import SortedList
from util import get_data
from sklearn.utils import shuffle
from datetime import datetime


class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_of_classes = len(set(Y))

    def predict(self, X):
        P = np.zeros(len(X))
        for j, x in enumerate(X):
            distances = SortedList()
            for i, point in enumerate(self.X):
                dist = np.linalg.norm(x - point)
                if len(distances) < self.k:
                    distances.add((dist, self.Y[i]))
                else:
                    if dist < distances[-1][0]:
                        distances.pop()
                        distances.add((dist, self.Y[i]))

            class_count = np.zeros(self.num_of_classes)
            for _, i in distances:
                class_count[i] += 1
            P[j] = np.argmax(class_count)

        return P

    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)


if __name__ == "__main__":
    X, Y = get_data(2000)
    Ntrain = 1000
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    for k in (1, 2, 3, 4, 5):
        knn = KNN(k)
        t0 = datetime.now()
        knn.fit(Xtrain, Ytrain)
        print "Training time:", datetime.now()-t0

        t0 = datetime.now()
        print "Train accuracy:", knn.score(Xtrain, Ytrain)
        print "Time t0 compute train accuracy:", (datetime.now(
        )-t0), "Train size:", len(Ytrain)

        t0 = datetime.now()
        print "Test accuracy:", knn.score(Xtest, Ytest)
        print "Time t0 compute test accuracy:", (datetime.now(
        ) - t0), "Test selfize:", len(Ytest)
