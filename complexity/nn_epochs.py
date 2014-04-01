"""
Plots Performance of Neural Networks with number of epochs
This is not for studying model complexity, but to see how long does it take for NN to converge
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

# We will test if the NN converges in 200 iterations
max_epochs = range(0, 2000)
train_err = zeros(len(max_epochs))
test_err = zeros(len(max_epochs))

# Build a network with 13 input nodes, 5 hidden nodes and 1 output nodes
# The networks is fully connected - a node from a layer is connected to all nodes 
# in its neighboring layer
net = buildNetwork(13, 5, 1)
# The dataset will have 13 features and 1 target label
ds = SupervisedDataSet(13, 1)

# Convert the boston dataset into SupervisedDataset
for i in range(1, len(X_train)):
    ds.addSample(X_train[i], y_train[i])

# Setup a trainer that will use backpropogation for training
trainer = BackpropTrainer(net, ds)

for i in max_epochs:
	# Run the backprop once
    train_err[i] = trainer.train()

    # Find the labels for test set
    y = zeros(len(X_test))
    for j in range(0, len(X_test)):
    	# Run X_test[j] on the NN and determine its label
        y[j] = net.activate(X_test[j])

    # Calculate MSE for all samples in the test set
    test_err[i] = mean_squared_error(y, y_test)

# Plot training and test error as a function of the number of epochs (iterations)
pl.figure()
pl.title('Neural Networks: Performance vs Num of Epochs')
pl.plot(max_epochs, test_err, lw=2, label = 'test error')
pl.plot(max_epochs, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Number of Epochs')
pl.ylabel('RMS Error')
pl.show()
