"""
Plots Model Complexity graphs for SVM
"""

import sys
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

clf = SVR(kernel='poly', degree=1)
clf.fit(X_train, y_train)

train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

print "Linear Kernel"
print train_err, test_err

clf = SVR(kernel='poly', degree=2)
clf.fit(X_train, y_train)

train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

print "Poly Kernel with degree 2"
print train_err, test_err

clf = SVR(kernel='rbf', degree=2)
clf.fit(X_train, y_train)

train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

print "RBF Kernel with degree 2"
print train_err, test_err

clf = SVR(kernel='rbf', degree=3)
clf.fit(X_train, y_train)

train_err = mean_squared_error(y_train, clf.predict(X_train))
test_err = mean_squared_error(y_test, clf.predict(X_test))

print "RBF Kernel with degree 3"
print train_err, test_err