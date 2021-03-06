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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(data, plot=False):
    data = np.load(data)
    if plot:
        l = data.shape[0]
        plot_3d(data[:, 0], data[:, 1], data[:, 2].reshape((l, 1)))
    return data


def get_pca(data):
    data = load_data(data)
    p = PCA(n_components=3)
    p.fit(data)
    return p


def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data_1 = 'pca1.npy'
    data_2 = 'pca2.npy'
    load_data(data_1, plot=True)
    load_data(data_2, plot=True)
    print('First dataset PCAs: {}'.format(get_pca(data_1).explained_variance_))
    print('Second dataset PCAs: {}'.format(get_pca(data_2).explained_variance_))
# ========================= EOF ================================================================
