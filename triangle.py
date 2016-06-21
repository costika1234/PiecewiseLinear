#/usr/local/bin/python

from cvxopt import matrix, solvers, sparse, spmatrix
from input_generator import InputGenerator
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from parser import Parser
from scipy.spatial import ConvexHull
from sympy import Plane, Point3D, poly, Line, Point2D
from utils import Utils

import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import re
import sympy as sp
import sys


def init_triangle_vertices():
    return [(2, 2), (4, 2), (2.5, 3.5)]


def init_function_info():
    return (2, 3.95)


def init_convex_polygon():
    # points = np.random.uniform(0.5, 2, 20).reshape((10, 2))
    # convex_hull = ConvexHull(points)
    # polygon_vertices = []

    # for vertex in convex_hull.vertices:
    #     polygon_vertices.append(tuple(convex_hull.points[vertex]))

    polygon_vertices = [
        (0.68057115927045464, 1.0814584085563212),
        (1.8806783285804756, 0.7398107495124151),
        (1.7257636277279738, 1.7320128263634973),
        (1.5760258289686251, 1.8376157951865417),
        (1.0106316531913753, 1.6809045490812751)]

    return polygon_vertices


def compute_plane_gradient(p1, p2, p3):
    # Equation of type: a x + b y + c z + d = 0.
    # Return gradient as 2D point: (-a / c, -b / c)
    plane_equation = Plane(Point3D(p1[0], p1[1], 'h1'),
                           Point3D(p2[0], p2[1], 'h2'),
                           Point3D(p3[0], p3[1], 'h3')).equation()

    plane_equation_poly = poly(str(plane_equation), gens=sp.symbols(['x', 'y', 'z']))
    a = plane_equation_poly.diff('x')
    b = plane_equation_poly.diff('y')
    c = plane_equation_poly.diff('z')

    return (- a / c, - b / c)


def compute_plane_gradient_2(p1, p2, p3):
    # Equation of type: a x + b y + c z + d = 0.
    # Return gradient as 2D point: (-a / c, -b / c)
    # p1, p2, p3 are Point3D objects.
    plane_equation = Plane(p1, p2, p3).equation()

    plane_equation_poly = poly(str(plane_equation), gens=sp.symbols(['x', 'y', 'z']))
    a = plane_equation_poly.diff('x')
    b = plane_equation_poly.diff('y')
    c = plane_equation_poly.diff('z')

    return (- a / c, - b / c)


def compute_point_coords(start, end, index):
    if index == 2:
        return Point3D(start[0], start[1], 'h2')
    elif index == 3:
        return Point3D(end[0], end[1], 'h3')

    return Point3D('(1 - a%d) * %f + a%d * %f' % (index, start[0], index, end[0]),
                   '(1 - a%d) * %f + a%d * %f' % (index, start[1], index, end[1]),
                   'h' + str(index))



class ConsistencyTriangle:

    def __init__(self):
        # Triangle vertices (2D coordinates).
        self.triangle = init_triangle_vertices()

        # Function information (pair of rational numbers).
        self.bounds = init_function_info()

        # Derivative information (convex polygon given by collection of vertices).
        self.polygon = init_convex_polygon()

        self.min_heights = np.zeros(3)
        self.max_heights = np.zeros(3)


    def solveLP(self):
        # Check simple case when gradient of plane is contained within the convex polygon.
        gradient = compute_plane_gradient(self.triangle[0], self.triangle[1], self.triangle[2])

        # Get all adjacent pairs of points that make up all sides of the polygon.
        points = zip(*[self.polygon, self.polygon[1:] + [self.polygon[0]]])

        # Vectors of coefficients.
        (h1_coefs, h2_coefs, h3_coefs, rhs_coefs) = ([], [], [], [])

        for (curr_point, next_point) in points:
            # Choose clockwise order when constructing line of eq. a x + b y + c = 0 from 2 given
            # points. Then the semi-plane containing the convex polygon will be given by:
            # a x + b y + c <= 0.
            line = Line(Point2D(next_point), Point2D(curr_point))
            line_equation_poly = poly(str(line.equation()), gens=sp.symbols(['x', 'y']))
            gradient_poly = poly(str(line_equation_poly.eval(gradient)),
                                 gens=sp.symbols(['h1', 'h2', 'h3']))

            h1_coefs.append(float(gradient_poly.diff('h1').eval((1, 0, 0))))
            h2_coefs.append(float(gradient_poly.diff('h2').eval((0, 1, 0))))
            h3_coefs.append(float(gradient_poly.diff('h3').eval((0, 0, 1))))
            rhs_coefs.append(-float(gradient_poly.eval((0, 0, 0))))

        # Construct LP problem.
        # Objective function: Minimise/Maximise h1 + h2 + h3
        objective_function_vector = matrix([1.0, 1.0, 1.0])

        # Coefficient matrix.
        ones_matrix = spmatrix([1.0, 1.0, 1.0], range(3), range(3))
        heights_coef_matrix = matrix([h1_coefs, h2_coefs, h3_coefs])
        bounds_coef_matrix = matrix([ones_matrix, -ones_matrix])
        coef_matrix = sparse([heights_coef_matrix, bounds_coef_matrix])

        # Column vector of right hand sides.
        column_vector = matrix(rhs_coefs + [self.bounds[1]] * 3 + [-self.bounds[0]] * 3)

        min_sol = solvers.lp(objective_function_vector, coef_matrix, column_vector)
        is_consistent = min_sol['x'] is not None

        if is_consistent:
            self.min_heights = np.array(min_sol['x'])
            print np.around(self.min_heights, decimals=2)

            max_sol = solvers.lp(-objective_function_vector, coef_matrix, column_vector)
            self.max_heights = np.array(max_sol['x'])
            print np.around(self.max_heights, decimals=2)
        else:
            # Try to determine triangulation of the given triangle.
            print "Triangulating..."

            # Consider N = # of polygon sides points along a side of the triangle.
            # Thus, we will introduce heights h4, h5, ..., h(N+3).
            points_along_side = []
            points_along_side.append(compute_point_coords(self.triangle[1], self.triangle[2], 2))

            for index in range(len(self.polygon)):
                points_along_side.append(compute_point_coords(self.triangle[1],
                                                              self.triangle[2], index + 4))

            points_along_side.append(compute_point_coords(self.triangle[1], self.triangle[2], 3))

            fixed_vertex = Point3D(self.triangle[0][0], self.triangle[0][1], 'h1')
            all_triangles = zip(*([fixed_vertex] * (1 + len(self.polygon)),
                                  points_along_side,
                                  points_along_side[1:]))

            all_gradients = [compute_plane_gradient_2(p1, p2, p3) for (p1, p2, p3) in all_triangles]

            for gradient in all_gradients:
                print gradient





    def plotDomain(self):
        # Triangle.
        triangle_vertices = self.triangle + [self.triangle[0]]
        triangle_codes = [Path.MOVETO] + [Path.LINETO] * 2 + [Path.CLOSEPOLY]
        triangle_path = Path(triangle_vertices, triangle_codes)

        # Polygon.
        polygon_vertices = self.polygon + [self.polygon[0]]
        polygon_codes = [Path.MOVETO] + [Path.LINETO] * (len(polygon_vertices) - 2) + \
                        [Path.CLOSEPOLY]
        polygon_path = Path(polygon_vertices, polygon_codes)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        triangle_patch = patches.PathPatch(triangle_path, facecolor='orange', lw=2)
        polygon_patch = patches.PathPatch(polygon_path, facecolor='green', lw=2)

        ax.add_patch(triangle_patch)
        ax.add_patch(polygon_patch)

        ax.set_xlim(-1, 4.5)
        ax.set_ylim(-1, 4.5)
        plt.show()


def main():
    cons = ConsistencyTriangle()
    cons.solveLP()
    # cons.plotDomain()


if __name__ == '__main__':
    main()
