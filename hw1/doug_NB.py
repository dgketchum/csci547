import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
iris = datasets.load_iris()
X = iris.data  # n x m matrix of features
y = iris.target  # n vector of classes
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33,
                                        random_state=42)  # Split into 33% test and 67% training sets
print(X.shape)
classes = [0, 1, 2]

m = X.shape[0]  # Number of data instances
m_test = X_test.shape[0]  # Number of test data instances
N = 3  # Number of classes
n = X.shape[1]  # Number of features

mu_array = np.zeros((n, N))
sigma2_array = np.zeros((n, N))
prior_array = np.zeros((N))

# Learning phase
for k in range(N):  # Loop over each class label
    C_k = classes[k]
    prior = sum(y == C_k) / float(y.shape[0])  # Count the number of data where the label is C_k
    mu = np.sum(X[y == C_k], axis=0) / len(
        X[y == C_k])  # Take the mean of those features where the corresponding label is C_k
    sigma2 = np.sum((X[y == C_k] - mu) ** 2, axis=0) / (
    len(X[y == C_k]) - 1)  # Take the variance of those features where the corresponding label is C_k
    mu_array[:, k] = mu  # Store in the arrays we created above
    sigma2_array[:, k] = sigma2
    prior_array[k] = prior

# Training set predictions
class_probabilities = np.zeros((m, N))  # The probabilities for

for i, x in enumerate(X):  # Loop over the training data instances
    for k in range(N):  # Loop over the classes
        prior = prior_array[k]
        mu = mu_array[:, k]
        sigma2 = sigma2_array[:, k]
        likelihood = np.prod(np.exp(-(x - mu) ** 2 / (2 * sigma2)))
        posterior_k = prior * likelihood
        class_probabilities[i, k] = posterior_k

class_probabilities /= np.sum(class_probabilities, axis=1, keepdims=True)
print(x, i)
print(class_probabilities[:, :])
# ========================= EOF ====================================================================
