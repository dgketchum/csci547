# =============================================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================================

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


def plot_gallery(n_comps=100):
    faces = fetch_lfw_people(resize=0.7, min_faces_per_person=50)
    x = faces.data
    cols, rows = 5, 2
    n_samples, h, w = faces.images.shape
    p = PCA(copy=True, n_components=n_comps, whiten=False)
    p.fit(x)
    eigenfaces = p.components_.reshape((n_comps, h, w))
    h, w = eigenfaces.shape[1], eigenfaces.shape[2]
    plt.figure(figsize=(1.8 * cols, 2.4 * rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
    plt.show()


def plot_face(comps):
    faces = fetch_lfw_people(resize=0.5, min_faces_per_person=50)
    x = faces.data
    n_samples, h, w = faces.images.shape

    p = PCA(copy=True, n_components=200, whiten=False)
    p.fit(x)
    X = p.transform(x)

    fig, axs = plt.subplots(nrows=3, ncols=2)
    data_index = 0
    for i, nc in enumerate(comps):
        x_rec = 0
        for c, l in zip(p.components_[:nc], X[data_index]):
            x_rec += c * l
        x_rec += p.mean_
        x_rec = x_rec.reshape((h, w))
        axs[i, 0].imshow(x[data_index, :].reshape((h, w)), cmap=plt.cm.gray)
        axs[i, 1].imshow(x_rec, cmap=plt.cm.gray)
    plt.show()


def face_SVM():
    faces = fetch_lfw_people(resize=0.5, min_faces_per_person=50)
    X = faces.data
    y = faces.target
    target_names = faces.target_names
    # n_classes = target_names.shape[0]
    # n_samples, h, w = faces.images.shape
    # n_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    p = PCA(copy=True, n_components=0.95, whiten=True).fit(X)

    X_train_pca = p.transform(X_train)
    X_test_pca = p.transform(X_test)

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    print('Accuracy: {}'.format(accuracy_score(y_test, y_pred)))
    print('Confusion matrix: \n{}'.format(confusion_matrix(y_test, y_pred)))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    plot_gallery(n_comps=100)
    plot_face(comps=[1, 10, 100])
    face_SVM()

# ========================= EOF ================================================================
