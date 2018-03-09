import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, normalize
from sklearn.metrics import confusion_matrix, accuracy_score


def softmax(X, W):
    a = np.dot(X, W)
    return np.exp(a) / np.repeat(np.sum(np.exp(a), axis=1, keepdims=True), N, axis=1)


def _J(X, W, T):
    return -np.sum(np.sum(T * np.log(softmax(X, W)), axis=1), axis=0)


def _gradient(X, W, T):
    return -np.column_stack([np.sum([(T - softmax(X, W))[i, k] * X[i] for i in range(m)], axis=0) for k in range(N)])


breast_cancer_data = load_breast_cancer()

x = breast_cancer_data.data
y = breast_cancer_data.target
N = x.shape[1]
D, D_test, y, y_test = train_test_split(x, y, test_size=0.33,
                                        random_state=42)
D = normalize(D)
D_test = normalize(D_test)

X = np.column_stack((np.ones_like(y), D))
m = X.shape[0]
n = X.shape[1]
N = breast_cancer_data.target_names.shape[0]

T = np.zeros((m, N))
for t, yi in zip(T, y):
    t[yi] = 1

N_iterations = 2000

eta = 0.001
W = 0.1 * np.random.randn(n, N)

for i in range(N_iterations):
    W -= eta * _gradient(X, W, T)

y_pred = np.argmax(softmax(X, W), axis=1)
print('Objective Function Value: ', _J(X, W, T),
      'Total misclassified: ', sum(y != y_pred))
print(confusion_matrix(y_pred, y))
print('Accuracy train: {}'.format(accuracy_score(y, y_pred)))

X = np.column_stack((np.ones_like(y_test), D_test))

y_test_pred = np.argmax(softmax(X, W), axis=1)

print(confusion_matrix(y_test_pred, y_test))
print('Accuracy test: {}'.format(accuracy_score(y_test, y_test_pred)))

# Using sklearn.preprocessing.scale
# Objective Function Value:  15.491380559972844 Total misclassified:  4
# [[142   1]
#  [  3 235]]
# 0.989501312335958
# [[ 65   1]
#  [  2 120]]
# 0.9840425531914894
#
# Using sklearn.preprocessing.normalize
# Objective Function Value:  147.42761335322936 Total misclassified:  44
# [[105   4]
#  [ 40 232]]
# 0.884514435695538
# [[ 55   1]
#  [ 12 120]]
# 0.9308510638297872


# ========================= EOF ====================================================================
