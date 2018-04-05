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
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def support_vector_machine(src, krnl='linear', on_loop=False, normal=False):
    csv = pd.read_csv(src)
    labels = csv.iloc[:, 0].values
    data = csv.iloc[:, 1:].values

    if normal:
        data = normalize(data)

    x, x_test, y, y_test = tts(data, labels, test_size=0.33)
    svc = SVC(kernel=krnl)
    svc.fit(x, y)
    y_pred = svc.predict(x)
    y_test_pred = svc.predict(x_test)
    training = sum(y == y_pred).astype(float) / len(y)
    testing = sum(y_test == y_test_pred).astype(float) / len(y_test_pred)
    if not on_loop:
        print("{} SVM Training accuracy with {}: {}, normalized {}".format(krnl, krnl, training, normal))
        print("{} SVM Test accuracy with {}: {}, normalized {}".format(krnl, krnl, testing, normal))
    return training, testing


def loop_svm(src, num_loops, krnl='linear', normed=False):
    iterlist = np.zeros((num_loops, 2))
    for i in range(num_loops):
        if normed:
            train, test = support_vector_machine(src, krnl=krnl, on_loop=True, normal=True)
        else:
            train, test = support_vector_machine(src, krnl=krnl, on_loop=True)
        iterlist[i] = [i, test]
    series = pd.Series(data=iterlist[:, 1], index=iterlist[:, 0])
    # plt.figure()
    # series.plot.hist(alpha=1)
    # plt.show()
    print('{} SVM Test mean: {}, test stdev: {} normalized {}'.format(krnl, iterlist[:, 1].mean(),
                                                                      iterlist[:, 1].std(), normed))
    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    source_data = 'wine.data'
    support_vector_machine(source_data, krnl='linear')
    loop_svm(source_data, 100, krnl='linear')
    loop_svm(source_data, 100, krnl='linear', normed=True)
    loop_svm(source_data, 100, krnl='poly')
    loop_svm(source_data, 100, krnl='poly', normed=True)
    loop_svm(source_data, 100, krnl='rbf')
    loop_svm(source_data, 100, krnl='rbf', normed=True)

# ========================= EOF ================================================================
