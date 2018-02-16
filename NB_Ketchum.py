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

# feature select

theta = np.zeros((n, N))

a = np.arange(0, n).reshape(8, 8)
b = a[:, 2:7].reshape(40)
X_mod = X[:, b]
for k in range(N):
    theta[:, k] = np.sum(X_mod[y == k], axis=0) / len(X_mod[y == k])

unique, counts = np.unique(y, return_counts=True)
priors = np.array([x / np.sum(counts) for x in counts])

class_probs = np.zeros((m, N))

for i, x in enumerate(X_mod):
    for k in range(N):
        prior = priors[k]
        lklhd = np.sum(theta[:, k] * (1 - theta[:, k]) ** (1 - x))
        pstrr_k = prior * lklhd
        class_probs[i, k] = pstrr_k
class_probs /= np.sum(class_probs, axis=1, keepdims=True)

y_pred_train = np.argmax(class_probs, axis=1)

cm_train = confusion_matrix(y_pred_train, y)

# print(cm_train)
# print(r)
print('training accuracy: {}'.format(accuracy_score(y, y_pred_train)))
# print('mean of features: {}'.format(np.mean(r)))


# ========================= EOF ====================================================================
