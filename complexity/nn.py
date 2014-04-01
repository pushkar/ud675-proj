"""
Plots Performance of Neural Networks when you change the network
We vary complexity by changing the number of hidden layers the network has
We use pybrain (http://pybrain.org/) to design and train our NN
"""

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

# List all the different networks we want to test again
# All networks have 13 input nodes and 1 output nodes
# All networks are fully connected
net = []
# 1 hidden layer with 1 node
net.append(buildNetwork(13,1,1))
# 1 hidden layer with 5 nodes
net.append(buildNetwork(13,5,1))
# 2 hidden layers with 7 and 3 nodes resp
net.append(buildNetwork(13,7,3,1))
# 3 hidden layers with 9, 7 and 3 nodes resp
net.append(buildNetwork(13,9,7,3,1))
# 4 hidden layers with 9, 7, 3 and 2 noes resp
net.append(buildNetwork(13,9,7,3,2,1))
net_arr = range(0, len(net))

# The dataset will have 13 features and 1 target label
ds = SupervisedDataSet(13, 1)

train_err = zeros(len(net))
test_err = zeros(len(net))

# We will train each NN for 50 epochs
max_epochs = 50

# Convert the boston dataset into SupervisedDataset
for j in range(1, len(X_train)):
	ds.addSample(X_train[j], y_train[j])

for i in range(1, len(net)):
	# Setup a trainer that will use backpropogation for training
	trainer = BackpropTrainer(net[i], ds)

	# Run backprop for max_epochs number of times
	for k in range(1, max_epochs):
		train_err[i] = trainer.train()

	# Find the labels for test set
	y = zeros(len(X_test))

	for j in range(0, len(X_test)):
		y[j] = net[i].activate(X_test[j])

    # Calculate MSE for all samples in the test set
	test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the number of hidden layers
pl.figure()
pl.title('Neural Networks: Performance vs Model Complexity')
pl.plot(net_arr, test_err, lw=2, label = 'test error')
pl.plot(net_arr, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Model Complexity')
pl.ylabel('RMS Error')
pl.show()
