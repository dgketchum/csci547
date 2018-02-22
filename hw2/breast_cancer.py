import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def _sigmoid(w, X):
    z = np.dot(X, w)
    return 1. / (1 + np.exp(-z))


def _J_fun(y, w, X):
    return -sum(y * np.log(_sigmoid(w, X)) + (1 - y) * np.log(1 - _sigmoid(w, X)))


def _gradient_fun(y, w, X):
    return np.dot(_sigmoid(w, X) - y, X)


breast_data = load_breast_cancer()

x = breast_data.data
y = breast_data.target
N = len(y)
X, X_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                        random_state=42)

N_iterations = 200000

eta = np.array([[0.1, 0.1], [0.0001, 0.01]])

w = np.array([-0.5, 0.5])

for i in range(N_iterations):
    grad_w = _gradient_fun(w, X)
    w -= np.dot(eta, grad_w)

classification_error = sum((_sigmoid(w, X) > 0.5) == y) / len(y)
print(classification_error)

pass

# ========================= EOF ====================================================================
