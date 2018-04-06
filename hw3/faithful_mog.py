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
from pandas import read_csv
from sklearn import mixture
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


def faithful(csv):
    csv = read_csv(csv, sep=',')
    data = csv.values
    mix = mixture.GaussianMixture(n_components=2)
    mix.fit(data)
    pred = mix.predict(data).reshape((272, 1))
    colors = ['r' if i == 0 else 'g' for i in pred]

    ndata = np.hstack((pred, data))
    one = ndata[ndata[:, 0] == 0]
    two = ndata[ndata[:, 0] == 1]
    pos_1 = (np.mean(one[:, 1]), np.mean(one[:, 2]))
    pos_2 = (np.mean(two[:, 1]), np.mean(two[:, 2]))

    cov_1 = np.cov(one[:, 1:])
    cov_2 = np.cov(two[:, 1:])

    ax = plt.gca()
    ax.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.8)

    plt.title('Old Faithful Data')
    plt.xlabel('Duration (min)')
    plt.ylabel('Interval (min)')
    plt.show()


def plot_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


if __name__ == '__main__':
    home = os.path.expanduser('~')
    data = 'faithful.dat'
    faithful(data)

# ========================= EOF ================================================================
