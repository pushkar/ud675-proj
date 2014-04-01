"""
Plots Model Complexity graphs for boosting, Adaboost in this case
For Boosting we vary model complexity by varying the number of base learners
"""

from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the number of base learners from 2 to 300
max_learners = arange(2, 300)
train_err = zeros(len(max_learners))
test_err = zeros(len(max_learners))

for i, l in enumerate(max_learners):
	# Set up a Adaboost Regression Learner with l base learners
    regressor = AdaBoostRegressor(n_estimators=l)

    # Fit the learner to the training data
    regressor.fit(X_train, y_train)

    # Find the MSE on the training set
    train_err[i] = mean_squared_error(y_train, regressor.predict(X_train))
    # Find the MSE on the testing set
    test_err[i] = mean_squared_error(y_test, regressor.predict(X_test))

# Plot training and test error as a function of the number of base learners
pl.figure()
pl.title('Boosting: Performance vs Number of Learners')
pl.plot(max_learners, test_err, lw=2, label = 'test error')
pl.plot(max_learners, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Learners')
pl.ylabel('RMS Error')
pl.show()