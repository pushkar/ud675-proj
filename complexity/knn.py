"""
Plots Model Complexity graphs for kNN
"""

import sys
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

k_range = arange(1, 30)
train_err = zeros(len(k_range))
crossval_err = zeros(len(k_range))

for i, k in enumerate(k_range):
    neigh = KNeighborsRegressor(n_neighbors=k)
    neigh.fit(X_train, y_train)

    train_err[i] = mean_squared_error(y_train, neigh.predict(X_train))
    crossval_err[i] = mean_squared_error(y_test, neigh.predict(X_test))

pl.figure()
pl.title('kNN: Error as a function of k')
pl.plot(k_range, crossval_err, lw=2, label = 'test error')
pl.plot(k_range, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('k')
pl.ylabel('RMS error')
pl.show()