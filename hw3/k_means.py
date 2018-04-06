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
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.datasets import load_sample_image


def get_flower(native=False, reduced=False):
    flower = load_sample_image('flower.jpg')
    if native:
        return flower
    if reduced:
        image = np.array(flower, dtype=np.float64) / 255.
        w, h, d = image.shape
        arr = np.reshape(image, (w * h, d))

        sample = shuffle(arr, random_state=0)[:1000]
        km = KMeans(n_clusters=8, random_state=0).fit(sample)
        labels = km.predict(arr)
        return km, labels, w, h


def plot_flower():
    plt.figure()
    plt.clf()
    ax = plt.axes([0, 0, 1, 1])
    plt.axis('off')
    plt.title('Original image')
    plt.imshow(get_flower(native=True))
    plt.show()


def plot_reduced_flower():
    km, labels, w, h = get_flower(reduced=True)
    plt.figure()
    plt.clf()
    plt.axis('off')
    plt.title('Quantized image')
    image = recreate_image(km.cluster_centers_, labels, w, h)
    plt.imshow(image)
    plt.show()


def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


if __name__ == '__main__':
    home = os.path.expanduser('~')
    plot_flower()
    plot_reduced_flower()

# ========================= EOF ================================================================
