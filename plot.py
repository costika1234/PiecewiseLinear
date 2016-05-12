import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


points = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3)]

x = [p[0] for p in points]
y = [p[1] for p in points]

print x
print y

triangles = [[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4], [3, 4, 6], [4, 7, 6], [4, 5, 7], [5, 8, 7], [6, 7, 9], [7, 10, 9], [7, 8, 10], [8, 11, 10]]

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(x, y, triangles=triangles)

z = [2] * 12

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap='BuGn', linewidth=1.0, antialiased=False, triangles=triangles)

ax.set_zlim(0, 2.5)

plt.show()