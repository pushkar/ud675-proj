"""
Bias and Variance
-----------------

This script plots some simple examples of how model complexity affects bias and variance.
"""

import sys
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

max_learners = arange(2, 300)
train_err = zeros(len(max_learners))
crossval_err = zeros(len(max_learners))

for i, l in enumerate(max_learners):
    regressor = AdaBoostRegressor(n_estimators=l)

    regressor.fit(X_train, y_train)

    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    crossval_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

pl.figure()
pl.title('Boosting: Performance vs Number of Learners')
pl.plot(max_learners, crossval_err, lw=2, label = 'test error')
pl.plot(max_learners, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Learners')
pl.ylabel('RMS Error')
pl.show()