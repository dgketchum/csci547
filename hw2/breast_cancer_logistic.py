import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def _sigmoid(w, X):
    z = np.dot(X, w)
    return 1. / (1 + np.exp(-z))


def _J_fun(y, w, X):
    return -sum(y * np.log(_sigmoid(w, X)) + (1 - y) * np.log(1 - _sigmoid(w, X)))


def _gradient_fun(w, X):
    return np.dot(_sigmoid(w, X) - y, X)


breast_cancer_data = load_breast_cancer()

x = breast_cancer_data.data
y = breast_cancer_data.target
N = x.shape[1]
D, D_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                        random_state=42)
D = scale(D)
D_test = scale(D_test)

X = np.column_stack((np.ones_like(y), D))
m = X.shape[0]
n = X.shape[1]
N = 2

N_iterations = 1000

eta = 0.001
w = 0.1 * np.random.randn(n)

for i in range(N_iterations):
    grad_w = _gradient_fun(w, X)
    w -= eta * grad_w

classification_error = sum((_sigmoid(w, X) > 0.5) == y) / len(y)
print(classification_error)

X_test = np.column_stack((np.ones_like(y_test), D_test))

classification_error_test = sum((_sigmoid(w, X_test) > 0.5) == y_test) / len(y_test)
print(classification_error_test)

# ========================= EOF ====================================================================
