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
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def get_faces(plot=False):
    faces = fetch_lfw_people(resize=0.7, min_faces_per_person=50)
    x = faces.data
    y = faces.target
    n_samples, h, w = faces.images.shape
    n_comps = 100
    p = PCA(n_components=n_comps)
    p.fit(x)
    eigenfaces = p.components_.reshape((n_comps, h, w))
    if plot:
        plot_gallery(eigenfaces, h, w)


def plot_gallery(images, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.xticks(())
        plt.yticks(())
    plt.show()


if __name__ == '__main__':
    home = os.path.expanduser('~')
    get_faces(plot=True)

# ========================= EOF ================================================================
