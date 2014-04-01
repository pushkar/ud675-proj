"""
Plots Model Complexity graphs for kNN
For kNN we vary complexity by chaning k
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will change k from 1 to 30
k_range = arange(1, 30)
train_err = zeros(len(k_range))
test_err = zeros(len(k_range))

for i, k in enumerate(k_range):
	# Set up a KNN model that regressors over k neighbors
    neigh = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the learner to the training data
    neigh.fit(X_train, y_train)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, neigh.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, neigh.predict(X_test))

# Plot training and test error as a function of k
pl.figure()
pl.title('kNN: Error as a function of k')
pl.plot(k_range, test_err, lw=2, label = 'test error')
pl.plot(k_range, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('k')
pl.ylabel('RMS error')
pl.show()