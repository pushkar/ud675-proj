"""
Loading the boston dataset and examining its target (label) distribution.

We draw the target distribution for the training and testing sets. 
Ideally they should look similar.
----------------
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle

# Load the boston dataset
boston = datasets.load_boston()

# Randomly shuffle the sample set. This data could be ordered and we want to shuffle 
# it so that we can divide it into training and testing set
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Plot the target distribution in training set
pl.figure()
pl.title('Training Set: MEDV distribution')
bins = range(0, 60, 5)
n, bins, patches = pl.hist(y_train, bins, normed=1, histtype='bar', rwidth=0.8)

# Plot the testing distribution in testing set
pl.figure()
pl.title('Test Set: MEDV distribution')
n, bins, patches = pl.hist(y_test, bins, normed=1, histtype='bar', rwidth=0.8)

pl.show()
