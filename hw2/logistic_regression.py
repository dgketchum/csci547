import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _sigmoid(w, X):
    z = np.dot(X, w)
    return 1. / (1 + np.exp(-z))


def _J_fun(y, w, X):
    return -sum(y * np.log(_sigmoid(w, X)) + (1 - y) * np.log(1 - _sigmoid(w, X)))


def _gradient_fun(w, X):
    return np.dot(_sigmoid(w, X) - y, X)


data = pd.read_csv('lobster_survive.dat', header=0, sep=r"\s{2,}", engine='python')
x = data['Len'].as_matrix().astype(float)
y = data['Survive'].as_matrix().astype(float)
N = len(y)

X = np.vander(x, 2, increasing=True)

eta = np.array([[0.000001, 0], [0, 0.000000001]])

N_iterations = 200000

w = np.array([-1., 0.5])

for i in range(N_iterations):
    grad_w = _gradient_fun(w, X)  # Compute the gradient of the objective function
    w -= np.dot(eta, grad_w)

classification_error = sum((_sigmoid(w, X) > 0.5) == y) / len(y)
print(classification_error)

# ========================= EOF ====================================================================
