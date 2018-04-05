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


def load_data(first, second, plot=False):
    f = np.load(first)
    s = np.load(second)
    dat = np.append(f, s, axis=0)
    if plot:
        plot_3d(dat[:, 0], dat[:, 1], dat[:, 2].reshape((42, 1)))


def get_pca(data, plot=False):
    p = PCA(data)
    if plot:
        plot_3d(data[:, 0], data[:, 1], data[:, 2].reshape((42, 1)))
    pass


def plot_3d(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x, y, z)
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data_1 = 'pca1.npy'
    data_2 = 'pca2.npy'
    load_data(data_1, data_2, plot=True)

# ========================= EOF ================================================================
