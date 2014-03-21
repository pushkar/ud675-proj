#!/usr/bin/env python
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

pl.figure()
pl.title('Training Set: MEDV distribution')
bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
n, bins, patches = pl.hist(y_train, bins, normed=1, histtype='bar', rwidth=0.8)

pl.figure()
pl.title('Test Set: MEDV distribution')
n, bins, patches = pl.hist(y_test, bins, normed=1, histtype='bar', rwidth=0.8)

pl.show()
