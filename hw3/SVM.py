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
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def support_vector_machine(src, krnl='linear'):
    csv = pd.read_csv(src)
    labels = csv.iloc[:, 0].values
    pre_data = csv.iloc[:, 1:].values
    data = minmax_scale(pre_data)
    x, x_test, y, y_test = tts(data, labels, test_size=0.33)
    svc = SVC(kernel=krnl)
    svc.fit(x, y)
    y_pred = svc.predict(x)
    y_test_pred = svc.predict(x_test)
    training = sum(y == y_pred).astype(float) / len(y)
    testing = sum(y_test == y_test_pred).astype(float) / len(y_test_pred)
    # print("Training accuracy with {}: {}".format(krnl, training))
    # print("Test accuracy with {}: {}".format(krnl, testing))
    return training, testing


def loop_svm(src, num_loops):
    iterlist = np.zeros((num_loops, 2))
    for i in range(num_loops):
        train, test = support_vector_machine(src)
        iterlist[i] = [i, test]
    series = pd.Series(data=iterlist[:, 1], index=iterlist[:, 0])
    series.plot()
    plt.figure()
    series.plot.hist(alpha=1)
    plt.show()
    return None


if __name__ == '__main__':
    home = os.path.expanduser('~')
    source_data = 'wine.data'
    support_vector_machine(source_data, krnl='linear')
    loop_svm(source_data, 100)
    support_vector_machine(source_data, krnl='poly')
    support_vector_machine(source_data, krnl='rbf')

# ========================= EOF ================================================================
