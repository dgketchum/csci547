import numpy as np
from hw2.neural_network import Network
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale
from pandas import get_dummies

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

m = mnist.train.images.shape[0]
n = mnist.train.images.shape[1]
N = len(np.unique(mnist.train.labels))

X = minmax_scale(mnist.train.images)
y = mnist.train.labels
T = get_dummies(y).values

X_test = mnist.test.images
y_test = mnist.test.labels

mnist = None

nn = Network([n, 300, N], [None, 'sigmoid', 'softmax'], [True, True, False],
             layer_weight_means_and_stds=[(0, 0.1), (0, 0.1)])

eta = 0.01
# Number of iterations to complete
N_iterations = 1000
cost = np.zeros((N_iterations, 2))
cost.fill(np.nan)

# Perform gradient descent
for i in range(N_iterations):

    # Run the features through the neural net (to compute a and z)
    y_pred = nn.feed_forward(X)

    # Compute the gradient
    grad_w = nn._gradient_fun(X, T)

    # Update the neural network weight matrices
    for w, gw in zip(nn.weights, grad_w):
        w -= eta * gw
        cost[i] = [i, nn._J_fun(X, T)]

    # Print some statistics every thousandth iteration
    if i % 100 == 0:
        misclassified = sum(np.argmax(y_pred, axis=1) != y.ravel())
        print("Iter: {0}, ObjFuncValue: {1:3f}, "
              "Misclassed: {2}".format(i, nn._J_fun(X, T),
                                       misclassified))

# Predict the training data and classify
y_pred = np.argmax(nn.feed_forward(X_test), axis=1)
print("Test data accuracy: {0:3f}".format(1 - sum(y_pred != y_test.ravel()) / float(len(y_test))))

plt.plot(cost[:, 0], cost[:, 1], 'r')
plt.xlabel('Iteration')
plt.ylabel('Cost Function Value')
plt.show()

# ========================= EOF ================================================================
