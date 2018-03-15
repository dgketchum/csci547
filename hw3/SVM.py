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
import pandas as pd


def support_vector_machine():
    csv = pd.read_csv('wine.data')
    labels = csv.iloc[:, 0].values
    pre_data = csv.iloc[:, 1:].values
    data = minmax_scale(pre_data)
    svc = SVC(kernel='linear')
    pass


if __name__ == '__main__':
    home = os.path.expanduser('~')
    support_vector_machine()

# ========================= EOF ================================================================
