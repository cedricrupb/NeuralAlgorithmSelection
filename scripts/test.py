import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import random
import math


def dist(x):
    return math.sqrt((x[0] - 1)**2 + (x[1] - 1)**2)

n = 500

X = np.ones((2, n)) + np.random.normal(0, 1, (2, n))
y = [(0.0, 0.8, 0.0, 0.7) if dist(X[:, i]) < 1 else (1.0, 0.0, 0.0, 0.7)
     for i in range(n)]

plt.subplot(121)
plt.scatter(X[0, :], X[1, :], c=y)

plt.subplot(122)
plt.scatter(X[0, :], (X[0, :] - 1)**2+(X[1, :] - 1)**2, c=y)

plt.show()
