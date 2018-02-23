import numpy as np
from neural_network import Network

m = 100
x = np.linspace(0, 1, m)
y = np.array([np.exp(-np.sin(4*np.pi*xx**3)) for xx in x])
N = 1
n = 1

eta = 0.001

nn = Network([n, 20, N], [None, 'identity', 'softmax'], [True, True, False],
             layer_weight_means_and_stds=[(0, 0.1), (0, 0.1)])

# ========================= EOF ================================================================
