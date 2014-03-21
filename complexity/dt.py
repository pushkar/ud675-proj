"""
Plots Model Complexity graphs for Decision Trees
"""

import sys
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

max_depth = arange(2, 25)
train_err = zeros(len(max_depth))
crossval_err = zeros(len(max_depth))
error = 1.0

for i, d in enumerate(max_depth):
    regressor = DecisionTreeRegressor(max_depth=d)

    regressor.fit(X_train, y_train)

    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    crossval_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

pl.figure()
pl.title('Decision Trees: Performance vs Max Depth')
pl.plot(max_depth, crossval_err, lw=2, label = 'test error')
pl.plot(max_depth, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('RMS Error')
pl.show()