"""
Plots Learning curves for Neural Networks
Plot performance of NN when we change the size of the training set
"""

import sys
from numpy import *
import pylab as pl
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

# Load the boston dataset and seperate it into training and testing set
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# We will vary the training set so that we have 10 different sizes
sizes = linspace(10, len(X_train), 10)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

# Build a network with 3 hidden layers
net = buildNetwork(13, 9, 7, 5, 1)
# The dataset will have 13 input features and 1 output
ds = SupervisedDataSet(13, 1)

for i,s in enumerate(sizes):
    # Populate the dataset for training
    ds.clear()
    for j in range(1, int(s)):
        ds.addSample(X_train[j], y_train[j])

    # Setup a backprop trainer
    trainer = BackpropTrainer(net, ds)

    # Train the NN for 50 epochs
    # The .train() function returns MSE over the training set
    for e in range(0, 50):
        train_err[i] = trainer.train()

    # Find labels for the test set
    y = zeros(len(X_test))
    for j in range(0, len(X_test)):
        y[j] = net.activate(X_test[j])

    # Find MSE for the test set
    test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Neural Networks: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()
