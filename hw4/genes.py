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
import pickle
import numpy as np

OBS_POSSIBLE = ['A', 'T', 'G', 'C']
OBS_CODE = [0, 1, 2, 3]
STATES_POSSIBLE = [0, 1]
EXAMPLE = 'TCTAGTCCAGATAATCTGGT'


def load_sequences(dat):
    with open(dat, 'rb') as input_data:
        data = pickle.load(input_data)
    return data


def get_markov_params(data):
    data = np.vstack((np.array(data[0]), np.array(data[1])))
    obs = np.array(data[0, :])
    states = np.array(data[1, :], dtype=int)

    priors = np.zeros((2, 4))
    A = np.zeros((2, 4, 4))

    for cls in STATES_POSSIBLE:
        obs_slice = ''.join(obs[states == cls])
        for j, base in enumerate(OBS_POSSIBLE):
            count = obs_slice.count(base)
            priors[cls, j] = count

        priors[cls, :] = priors[cls, :] / np.sum(priors[cls, :])

        if abs(np.sum(priors[cls, :]) - 1.) > 0.001:
            raise Exception

        code_slice = [OBS_POSSIBLE.index(base) for base in obs_slice]
        previous = np.random.choice(OBS_CODE)
        for item in code_slice:
            A[cls, previous, item] += 1
            previous = item

        for row in range(A.shape[1]):
            A[cls, row, :] = A[cls, row, :] / np.sum(A[cls, row, :])
            if abs(np.sum(A[cls, row, :]) - 1) > 0.001:
                raise Exception

    return A, priors


def generate_sequence(transition_matrix, prior, k=0):
    A = transition_matrix[k]
    prior = prior[k]

    sequence = list(EXAMPLE)
    base = np.random.choice(OBS_POSSIBLE, replace=True, p=prior)
    sequence[0] = base

    prev_base = OBS_POSSIBLE.index(base)
    for i, letter in enumerate(sequence[1:], start=1):
        sequence[i] = np.random.choice(OBS_POSSIBLE, replace=True, p=A[prev_base])
        prev_base = OBS_POSSIBLE.index(letter)
    sequence = ''.join(sequence)

    return sequence


if __name__ == '__main__':
    home = os.path.expanduser('~')
    training_data = load_sequences('genes_training.p')
    transition, prior_matrix = get_markov_params(training_data)
    generate_sequence(transition, prior_matrix, k=1)

# ========================= EOF ================================================================
