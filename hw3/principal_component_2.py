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


def get_faces(n_comps):
    faces = fetch_lfw_people(resize=0.7, min_faces_per_person=50)
    x = faces.data
    n_samples, h, w = faces.images.shape
    p = PCA(n_components=n_comps)
    p.fit(x)
    eigenfaces = p.components_.reshape((n_comps, h, w))
    return eigenfaces


def plot_gallery(images, rows, cols):
    h, w = images.shape[1], images.shape[2]
    plt.figure(figsize=(1.8 * cols, 2.4 * rows))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    ef = get_faces(n_comps=100)
    plot_gallery(ef, 2, 5)
    ef = get_faces(n_comps=1)
    plot_gallery(ef, 1, 1)
    ef = get_faces(n_comps=10)
    plot_gallery(ef, 2, 5)
    df = get_faces(n_comps=100)
    plot_gallery(ef, 10, 10)
# ========================= EOF ================================================================
