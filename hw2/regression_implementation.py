import numpy as np
from neural_network import Network

m = 100
X = np.linspace(0, 1, m).reshape((m, 1))
y = np.array([np.exp(-np.sin(4 * np.pi * xx ** 3)) for xx in X]).reshape((m, 1))
N = 1
n = 1

nn = Network([n, 20, N], [None, 'identity', 'identity'], [True, True, False],
             layer_weight_means_and_stds=None)

eta = 0.001
# Number of iterations to complete
N_iterations = 10000

# Perform gradient descent
for i in range(N_iterations):

    # For stochastic gradient descent, take random samples of X and T

    # Run the features through the neural net (to compute a and z)
    y_pred = nn.feed_forward(X)

    # Compute the gradient
    grad_w = nn._gradient_fun(X, y)

    # Update the neural network weight matrices
    for w, gw in zip(nn.weights, grad_w):
        w -= eta * gw

    # Print some statistics every thousandth iteration
    if i % 1000 == 0:
        misclassified = sum(np.argmax(y_pred, axis=1) != y.ravel())
        print("Iteration: {0}, Objective Function Value: {1:3f}, "
              "Misclassified: {2}".format(i, nn._J_fun(X, y),
                                          misclassified))

pass
# ========================= EOF ================================================================
