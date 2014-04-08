#!/bin/python
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

def cross_val(x, y, d):
    f = polyfit(x_train, y_train, d)


y_train = array([1.0, 2.0, 4.5, 5.0, 5.1, 5.5, 6.0])
x_train = array([1000, 1500, 3500, 4000, 5500, 8500, 10000])

y_test = array([3.0, 5.5])
x_test = array([2000, 7000])

n = len(x_train)
degrees = linspace(1, n, n)
train_err = zeros(n)
test_err = zeros(n)

pl.figure()
pl.title('Housing Example: Polynomial Regression Example')
pl.plot(x_train, y_train, 'bo')
pl.plot(x_test, y_test, 'b*')
print x_test
for i, d in enumerate(degrees):
    f = polyfit(x_train, y_train, d)
    train_err[i] = mean_squared_error(y_train, polyval(f, x_train))
    test_err[i] = mean_squared_error(y_test, polyval(f, x_test))
    xfit = linspace(500, 10500, 500)
    yfit = polyval(f, xfit)
    pl.plot(xfit, yfit, lw=2, label = 'd =' + str(d))
pl.legend()
pl.xlabel('x')
pl.ylabel('y')

# Plot training and test error as a function of the training size
pl.figure()
pl.title('Housing Example: Performance vs Model Complexity')
pl.plot(degrees, train_err, lw=2, label = 'train error')
pl.plot(degrees, test_err, lw=2, label = 'test error')
pl.legend()
pl.xlabel('Degree Polynomial')
pl.ylabel('RMS Error')
pl.show()
