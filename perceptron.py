import numpy as np
import scipy.stats as stats

import numpy as np
from sortedcontainers import SortedList
from util import get_data, get_data_other
from sklearn.utils import shuffle
from datetime import datetime


class perceptron(object):
    def __init__(self):
        pass

    def fit(self, X, Y, eta=0.1, epochs=100):
        N, D = X.shape
        X = np.append(X, np.ones((N, 1)), axis=1)
        self.W = np.random.randn(D+1)
        for i in range(epochs):
            p = self.predict(X)
            X_miscl_indices = np.nonzero(Y != p)[0]
            if len(X_miscl_indices) == 0:
                break
            j = np.random.choice(X_miscl_indices)
            self.W += eta * Y[j] * X[j]

    def predict(self, X):
        p = X.dot(self.W)
        return np.sign(p)

    def score(self, X, Y):
        p = self.predict(np.append(X, np.ones((len(X), 1)), axis=1))
        return np.mean(p == Y)


if __name__ == "__main__":
    X, Y = get_data_other()
    X, Y = shuffle(X, Y)
    Y[Y == 0] = -1
    Ntrain = 3*int(len(X)/4)
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    model = perceptron()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print "Training time:", datetime.now()-t0

    t0 = datetime.now()
    print "Train accuracy:", model.score(Xtrain, Ytrain)
    print "Time t0 compute train accuracy:", (datetime.now(
    )-t0), "Train size:", len(Ytrain)

    t0 = datetime.now()
    print "Test accuracy:", model.score(Xtest, Ytest)
    print "Time t0 compute test accuracy:", (datetime.now(
    ) - t0), "Test selfize:", len(Ytest)
