import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def sigmoid(X, w):
    a = np.dot(X, w)
    return np.exp(a) / np.repeat(np.sum(np.exp(a), axis=1, keepdims=True), N, axis=1)


def _I(w):
    w = w.reshape((n, N))
    return -np.sum(np.sum(T * np.log(sigmoid(X, w)), axis=1), axis=0)


def _J(w):
    w = w.reshape((n, N))
    return -np.column_stack(
        [np.sum([(T - sigmoid(X, w))[i, k] * X[i] for i in range(m)], axis=0) for k in range(N)])  # .ravel()


data = pd.read_csv('./train.csv', index_col=0)


def make_one_hot(x):
    m = len(x)
    C = len(np.unique(x))
    V_1hot = np.zeros((m, C))
    for t, yi in zip(V_1hot, x):
        t[yi] = 1
    return V_1hot


def make_data_matrix(data):
    data = data.drop('Cabin', axis=1)
    data = data.drop('Ticket', axis=1)
    data = data.drop('Name', axis=1)

    data = data.dropna()

    dict_sex = {'male': 0, 'female': 1}
    dict_emb = {'S': 0, 'C': 1, 'Q': 2}

    data.Sex = data.Sex.map(dict_sex)
    data.Embarked = data.Embarked.map(dict_emb)

    y = data['Survived'].as_matrix().astype(int)
    data = data.drop('Survived', axis=1)

    m = len(y)

    Pclass_1hot = make_one_hot(data['Pclass'] - 1)
    Sex_1hot = make_one_hot(data['Sex'])
    SibSp_1hot = make_one_hot(data['SibSp'])
    Parch_1hot = make_one_hot(data['Parch'])
    Embarked_1hot = make_one_hot(data['Embarked'])
    X = np.column_stack(
        (np.ones(len(y)), Pclass_1hot, data['Age'], data['SibSp'], data['Parch'], data['Fare'], Embarked_1hot))
    X[:, 1:] = (X[:, 1:] - X[:, 1:].min()) / (X[:, 1:].max(axis=0) - X[:, 1:].min(axis=0))
    return m, X, y


m, X, y = make_data_matrix(data)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

classes = [0, 1]

m = len(y)
N = len(classes)
n = X.shape[1]

T = np.zeros((m, N))
for t, yi in zip(T, y):
    t[yi] = 1

w = np.random.randn(n, N)
c = 0
N_iterations = 10000
eta = 0.001
for l in range(N_iterations):
    w -= eta * _J(w)
    y_pred = np.argmax(sigmoid(X, w), axis=1)
    print(_I(w), sum(y != y_pred) / float(len(y)))
    c += 1

y_pred = np.argmax(sigmoid(X_test, w), axis=1)
pred_error = sum((y_test != y_pred)) / float(len(y_test))
print(pred_error)


# ========================= EOF ====================================================================
