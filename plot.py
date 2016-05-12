#/usr/local/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

x = [0, 1, 1, 0]
y = [0, 0, 1, 1]
z = [2, 1, 2, 1]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap='BuGn', linewidth=1.0, antialiased=False)

plt.show()
