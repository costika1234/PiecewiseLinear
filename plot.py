#/usr/local/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

#X = np.array([0.0, 1.0])
#Y = np.array([0.0, 1.0])

#X, Y = np.meshgrid(X, Y)

#Z = np.array([[1, 4], [9, 16]])

#fig = plt.figure()

#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='CMRmap', linewidth=0, antialiased=False)
#ax.set_zlim(0,16)

#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()


x = [0,1,1,0]
y = [0,0,1,1]
z = [0,1,0,1]

verts = [zip(x, y,z)]

print verts

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_trisurf(x, y, z, cmap='BuGn', linewidth=1.0, antialiased=False)

plt.show()
