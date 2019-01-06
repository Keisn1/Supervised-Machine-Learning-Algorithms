import numpy as np
import scipy.stats as stats

import numpy as np
from sortedcontainers import SortedList
from util import get_data
from sklearn.utils import shuffle
from datetime import datetime


class bayes(object):
    def __init__(self):
        pass

    def fit(self, X, Y, smoothing=1e-3):
        D = X.shape[1]
        self.K = len(set(Y))
        self.Gaussians = {}
        self.priors = {}
        for cl in range(self.K):
            X_cl = X[Y == cl]
            mu_cl = np.mean(X_cl, axis=0)
            cov_cl = np.cov(X_cl, rowvar=False) + np.eye(D) + smoothing
            self.Gaussians[cl] = {'mu': mu_cl, 'cov': cov_cl}
            self.priors[cl] = np.mean(Y == cl)

    def predict(self, X):
        P = np.zeros(len(X))
        posteriors = np.zeros((len(X), self.K))
        for cl in range(self.K):
            mean = self.Gaussians[cl]['mu']
            cov = self.Gaussians[cl]['cov']
            prior = self.priors[cl]
            logs = stats.multivariate_normal.logpdf(X, mean, cov)
            posteriors[:, cl] = logs + np.log(prior)
        return np.argmax(posteriors, axis=1)

    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)


if __name__ == "__main__":
    X, Y = get_data(10000)
    Ntrain = int(len(X)/2)
    Xtrain, Ytrain = X[:Ntrain, :], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:, :], Y[Ntrain:]

    model = naive_bayes()
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
