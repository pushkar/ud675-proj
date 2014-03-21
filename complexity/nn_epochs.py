"""
Plots Performance of Neural Networks with number of epochs
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

boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

max_epochs = range(0, 2000)

train_err = zeros(len(max_epochs))
test_err = zeros(len(max_epochs))

net = buildNetwork(13, 5, 1)
ds = SupervisedDataSet(13, 1)

for i in range(1, len(X_train)):
    ds.addSample(X_train[i], y_train[i])

trainer = BackpropTrainer(net, ds)

for i in max_epochs:
    train_err[i] = trainer.train()

    y = zeros(len(X_test))
    for j in range(0, len(X_test)):
        y[j] = net.activate(X_test[j])

    test_err[i] = mean_squared_error(y, y_test)

pl.figure()
pl.title('Neural Networks: Performance vs Num of Epochs')
pl.plot(max_epochs, test_err, lw=2, label = 'test error')
pl.plot(max_epochs, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Epochs')
pl.ylabel('RMS Error')
pl.show()
