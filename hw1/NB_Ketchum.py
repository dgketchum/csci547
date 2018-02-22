import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = np.round(digits.data / 16.)
y_classed = digits.target
target_arr = digits.target_names

X, X_test, y, y_test = train_test_split(X, y_classed, test_size=0.33,
                                        random_state=42)

m = X.shape[0]
n = X.shape[1]
N = target_arr.shape[0]
m_test = X_test.shape[1]

theta = np.zeros((n, N))
for k in range(N):
    theta[:, k] = np.sum(X[y == k], axis=0) / len(X[y == k])

unique, counts = np.unique(y, return_counts=True)
priors = np.array([x / np.sum(counts) for x in counts])

class_probs = np.zeros((m, N))

for i, x in enumerate(X):
    for k in range(N):
        prior = priors[k]
        lklhd = np.prod(theta[:, k] ** x * (1 - theta[:, k]) ** (1 - x))
        pstrr_k = prior * lklhd
        class_probs[i, k] = pstrr_k

class_probs /= np.sum(class_probs, axis=1, keepdims=True)
y_pred_train = np.argmax(class_probs, axis=1)
cm_train = confusion_matrix(y_pred_train, y)

# print(cm_train)

# print(r)
train_accuracy = accuracy_score(y, y_pred_train)
print('training accuracy: {}, trying to beat 0.913549459684123'.format(
    train_accuracy))

class_probs_test = np.zeros((m, N))

for i, xt in enumerate(X_test):
    for k in range(N):
        prior = priors[k]
        lklhd = np.prod(theta[:, k] ** xt * (1 - theta[:, k]) ** (1 - xt))
        pstrr_k = prior * lklhd
        class_probs_test[i, k] = pstrr_k

class_probs_test /= np.sum(class_probs_test, axis=1, keepdims=True)
y_pred_test = np.argmax(class_probs_test, axis=1)
cm_test = confusion_matrix(y_pred_test, y_test)

test_accuracy = accuracy_score(y_test, y_pred_test)


# ========================= EOF ====================================================================
