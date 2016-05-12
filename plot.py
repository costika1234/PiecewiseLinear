import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri


points = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
x = [p[0] for p in points]
y = [p[1] for p in points]

triangles = np.asarray([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])

# Triangulate parameter space to determine the triangles
tri = mtri.Triangulation(x, y, triangles=triangles)

z = [1, 2, 1.3, 1.5, 1.5, 2]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x, y, z, cmap='BuGn', linewidth=1.0, antialiased=False, triangles=triangles)

plt.show()