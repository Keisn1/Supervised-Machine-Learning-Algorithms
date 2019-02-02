from util import get_donut, get_xor, get_data_for_tree, get_data_other, get_data
import numpy as np
import matplotlib.pyplot as plt


class tree_node(object):
    def __init__(self, depth, max_depth):
        self.depth = depth
        self.max_depth = max_depth
        self.condition = None
        self.column_idx = None
        self.left_node = None
        self.right_node = None
        self.left_prediction = None
        self.right_prediction = None

    def print_node(self):
        print self.depth
        print "columns_idx and condition ", self.column_idx, self.condition
        print "left_prediction and right_prediction", self.left_prediction, self.right_prediction

        if self.left_node is not None:
            self.left_node.print_node()
        if self.right_node is not None:
            self.right_node.print_node()

        if self.left_node is None and self.right_node is None:
            print "end_note"
            print "left_prediction and right_prediction", self.left_prediction, self.right_prediction

    def predict_one(self, x):
        # print self.depth, self.left_prediction, self.right_prediction
        if x[self.column_idx] < self.condition:
            if self.left_node is not None:
                return self.left_node.predict_one(x)
            else:
                return self.left_prediction
        else:
            if self.right_node is not None:
                return self.right_node.predict_one(x)
            else:
                return self.right_prediction

    def fit(self, X, Y):
        best_IG = 0             # best information gain
        best_feature = 0        # corresponding best attribute
        best_column_idx = 0
        best_condition = 0
        for column_idx, feature in enumerate(X.T):
            # looks for the best split of the feature
            IG, condition = find_the_best_split(feature, Y)
            if IG > best_IG:
                best_IG = IG
                best_feature = feature
                best_column_idx = column_idx
                best_condition = condition

        # create subsets
        self.condition = best_condition
        self.column_idx = best_column_idx
        N = len(Y)
        idx_bool_left = best_feature < best_condition
        idx_bool_right = best_feature >= best_condition
        X_left, Y_left = X[idx_bool_left], Y[idx_bool_left]
        X_right, Y_right = X[idx_bool_right], Y[idx_bool_right]

        # base cases
        if self.depth == self.max_depth:
            self.left_prediction = np.round(Y_left.mean())
            self.right_prediction = np.round(Y_right.mean())

        if len(Y_left) == 0 or len(Y_right) == 0:
            self.left_prediction = self.right_prediction = np.round(Y.mean())
        else:
            if len(frozenset(Y_left)) == 1:
                self.left_prediction = Y_left[0]
            else:
                self.left_node = tree_node(self.depth+1, self.max_depth)
                self.left_node.fit(X_left, Y_left)
            if len(frozenset(Y_right)) == 1:
                self.right_prediction = Y_right[0]
            else:
                self.right_node = tree_node(self.depth+1, self.max_depth)
                self.right_node.fit(X_right, Y_right)


class decision_tree(object):
    def __init__(self, max_depth):
        self.root = tree_node(0, max_depth)

    def fit(self, X, Y):
        self.root.fit(X, Y)

    def print_tree(self):
        self.root.print_node()

    def predict(self, X):
        predictions = np.array([])
        for x in X:
            predictions = np.append(predictions, self.root.predict_one(x))
        return predictions

    def score(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)


def get_most_frequent_value(array):
    counts = np.bincount(array)
    return np.argmax(counts)


def entropy(Y):
    labels = set(Y)
    entropy = 0
    for label in labels:
        p_label = len(Y[Y == label])/float(len(Y))
        entropy -= p_label * np.log2(p_label)
    return entropy


def find_the_best_split(feature, Y):
    idx_sort = np.argsort(feature)
    feature = feature[idx_sort]
    Y = Y[idx_sort]
    current_val = Y[0]
    best_idx = 0
    best_IG = -1
    for idx in range(len(Y)):
        if current_val != Y[idx]:
            Y_left, Y_right = Y[:idx], Y[idx:]
            p_left = len(Y_left)/float(len(Y))
            IG = entropy(Y) - (p_left * entropy(Y_left) +
                               (1 - p_left) * entropy(Y_right))
            current_val = Y[idx]
            if IG > best_IG:
                best_IG = IG
                best_idx = idx
    return best_IG, (feature[best_idx-1] + feature[best_idx])/2.


def main():
    X, Y = get_data(10000)
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    N = len(X)
    print N
    N_train = int(0.7 * N)
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = decision_tree(7)
    model.fit(X_train, Y_train)
    print model.score(X_test, Y_test)


main()
