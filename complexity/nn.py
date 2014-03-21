"""
Plots Performance of Neural Networks when you change the network
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

net = []
net.append(buildNetwork(13,1,1))
net.append(buildNetwork(13,5,1))
net.append(buildNetwork(13,7,3,1))
net.append(buildNetwork(13,9,7,3,1))
net.append(buildNetwork(13,9,7,3,2,1))
net_arr = range(0, len(net))
ds = SupervisedDataSet(13, 1)
max_epochs = 50

train_err = zeros(len(net))
test_err = zeros(len(net))

for i in range(1, len(net)):
	ds.clear()

	for j in range(1, len(X_train)):
		ds.addSample(X_train[j], y_train[j])

	trainer = BackpropTrainer(net[i], ds)

	for k in range(1, max_epochs):
		train_err[i] = trainer.train()

	y = zeros(len(X_test))

	for j in range(0, len(X_test)):
		y[j] = net[i].activate(X_test[j])

	test_err[i] = mean_squared_error(y, y_test)

pl.figure()
pl.title('Neural Networks: Performance vs Model Complexity')
pl.plot(net_arr, test_err, lw=2, label = 'test error')
pl.plot(net_arr, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Model Complexity')
pl.ylabel('RMS Error')
pl.show()
