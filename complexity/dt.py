"""
Plots Model Complexity graphs for Decision Trees
For Decision Trees we vary complexity by changing the size of the decision tree
"""

from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the depth of decision trees from 2 to 25
max_depth = arange(2, 25)
train_err = zeros(len(max_depth))
test_err = zeros(len(max_depth))

for i, d in enumerate(max_depth):
	# Setup a Decision Tree Regressor so that it learns a tree with depth d
    regressor = DecisionTreeRegressor(max_depth=d)
    
    # Fit the learner to the training data
    regressor.fit(X_train, y_train)

	# Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the depth of the decision tree learnt
pl.figure()
pl.title('Decision Trees: Performance vs Max Depth')
pl.plot(max_depth, test_err, lw=2, label = 'test error')
pl.plot(max_depth, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Max Depth')
pl.ylabel('RMS Error')
pl.show()