import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from datetime import datetime


def softmax(X, W, N):
    a = np.dot(X, W)
    return np.exp(a) / np.repeat(np.sum(np.exp(a), axis=1, keepdims=True), N, axis=1)


def _J(X, W, T, N):
    return -np.sum(np.sum(T * np.log(softmax(X, W, N)), axis=1), axis=0)


def _gradient(X, W, T, m, N):
    return -np.column_stack([np.sum([(T - softmax(X, W, N))[i, k] * X[i] for i in range(m)],
                                    axis=0) for k in range(N)])


first = True
for csv in ['titanic_train.csv', 'titanic_test.csv']:

    df = pd.read_csv(csv, engine='python')
    df.drop(columns=['PassengerId', 'Ticket', 'Cabin', 'Name',
                     'SibSp', 'Parch'], inplace=True)
    df.dropna(axis=0, how='any', inplace=True)

    one_hot_embark = pd.get_dummies(df['Embarked'])
    one_hot_sex = pd.get_dummies(df['Sex'])
    one_hot_pclass = pd.get_dummies(df['Pclass'])
    one_hot_pclass.columns = ['c1', 'c2', 'c3']

    df.drop(columns=['Sex', 'Embarked', 'Pclass'], inplace=True)
    df = df.join([one_hot_pclass, one_hot_sex, one_hot_embark], how='outer')

    if first:
        y = df['Survived'].values
        train = df.drop(columns=['Survived'])
        x = train.values
        min_max_scaler = MinMaxScaler()
        D = min_max_scaler.fit_transform(x)
        first = False
    else:
        y_test = df['Survived'].values
        test = df.drop(columns=['Survived'])
        x = test.values
        min_max_scaler = MinMaxScaler()
        D_test = min_max_scaler.fit_transform(x)

N = train.shape[1]

X = np.column_stack((np.ones_like(y), D))
m = X.shape[0]
n = X.shape[1]

T = np.zeros((m, N))
for t, yi in zip(T, y):
    t[yi] = 1

N_iterations = 1000

eta = 0.001
W = 0.1 * np.random.randn(n, N)

cost_funct = np.zeros((N_iterations, 2))
for i in range(N_iterations):
    W -= eta * _gradient(X, W, T, m, N)
    cost_funct[i] = [i, _J(X, W, T, N)]
    if i in [1, 100, 500, 1000, 5000]:
        print('Iteration {} at {}'.format(i, datetime.strftime(datetime.now(), '%H:%M:%S')))

y_pred = np.argmax(softmax(X, W, N), axis=1)
print('Objective Function Value: ', _J(X, W, T, N),
      'Total misclassified: ', sum(y != y_pred))
print(confusion_matrix(y_pred, y))
print(accuracy_score(y, y_pred))

X = np.column_stack((np.ones_like(y_test), D_test))

y_test_pred = np.argmax(softmax(X, W, N), axis=1)

print(confusion_matrix(y_test_pred, y_test))
print(accuracy_score(y_test, y_test_pred))

plt.plot(cost_funct[:, 0], cost_funct[:, 1], 'r')
plt.xlabel('Iteration')
plt.ylabel('Cost Function')
plt.show()

# ========================= EOF ================================================================
