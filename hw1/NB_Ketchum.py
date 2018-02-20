# =============================================================================================
# Copyright 2017 dgketchum
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


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = np.round(digits.data / 16.)
y_classed = digits.target
target_arr = digits.target_names

X, X_test, y, y_test = train_test_split(X, y_classed, test_size=0.33,
                                        random_state=42)

m = X.shape[0]
n = X.shape[1]
N = target_arr.shape[0]
m_test = X_test.shape[1]

train_acc = 0.0
# feature_select
list_ = [x for x in range(1, n)]
_list = list_
for start in _list:
    for stop in list_:

        n = abs(stop - start)
        X_mod = X[:, start: stop]
        theta = np.zeros((n, N))
        if X_mod.shape[1] > 1:
            for k in range(N):
                theta[:, k] = np.sum(X_mod[y == k], axis=0) / len(X_mod[y == k])

            unique, counts = np.unique(y, return_counts=True)
            priors = np.array([x / np.sum(counts) for x in counts])

            class_probs = np.zeros((m, N))

            for i, x in enumerate(X_mod):
                for k in range(N):
                    prior = priors[k]
                    lklhd = np.prod(theta[:, k] ** x * (1 - theta[:, k]) ** (1 - x))
                    pstrr_k = prior * lklhd
                    class_probs[i, k] = pstrr_k

            class_probs /= np.sum(class_probs, axis=1, keepdims=True)

            y_pred_train = np.argmax(class_probs, axis=1)

            cm_train = confusion_matrix(y_pred_train, y)

            # print(cm_train)
            # print(r)
            trial_accuracy = accuracy_score(y, y_pred_train)
            if trial_accuracy > train_acc:
                print('training accuracy: {}'.format(trial_accuracy))
                arr = X_mod[0, :]
                train_acc = trial_accuracy
                results = {'accuracy': train_acc, 'feature_array': arr}

print(results)

# ========================= EOF ====================================================================
