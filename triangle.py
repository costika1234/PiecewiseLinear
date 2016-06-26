#/usr/local/bin/python

from cvxopt import matrix, solvers, sparse, spmatrix
from input_generator import InputGenerator
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from parser import Parser
from scipy.spatial import ConvexHull
from sympy import Plane, Point3D, poly, Line, Point2D, Segment, Polygon
from utils import Utils

import itertools
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import re
import sympy as sp
import sys


def init_triangle_vertices():
    return [(0, 0), (1, 1), (-1, 1)]


def init_function_info():
    return (2, 10)


def init_convex_polygon():
    # Return polygon vertices in counter-clockwise order.
    points = np.random.uniform(1, 3, 20).reshape((10, 2))
    convex_hull = ConvexHull(points)
    polygon_vertices = []

    for vertex in convex_hull.vertices:
         polygon_vertices.append(tuple(convex_hull.points[vertex]))

    return polygon_vertices


def compute_plane_gradient(p1, p2, p3):
    # Equation of type: a x + b y + c z + d = 0.
    # Return gradient as 2D point: (-a / c, -b / c)
    # p1, p2, p3 are Point3D objects.
    plane_equation = Plane(p1, p2, p3).equation()

    plane_equation_poly = poly(str(plane_equation), gens=sp.symbols(['x', 'y', 'z']))
    a = plane_equation_poly.diff('x')
    b = plane_equation_poly.diff('y')
    c = plane_equation_poly.diff('z')

    return (- a / c, - b / c)


class ConsistencyTriangle:

    def __init__(self):
        # Triangle vertices (2D coordinates).
        self.triangle = init_triangle_vertices()

        # Function information (pair of rational numbers).
        self.bounds = init_function_info()

        # Derivative information (convex polygon given by collection of vertices).
        self.polygon = init_convex_polygon()

        self.polygon_lines = []
        self.adj_polygon_points = []

        self.min_heights = np.zeros(3)
        self.max_heights = np.zeros(3)


    def solveLP(self):
        # Check simple case when gradient of plane is contained within the convex polygon.
        gradient = compute_plane_gradient(Point3D(self.triangle[0][0], self.triangle[0][1], 'h0'),
                                          Point3D(self.triangle[1][0], self.triangle[1][1], 'h1'),
                                          Point3D(self.triangle[2][0], self.triangle[2][1], 'h2'))

        # Get all adjacent pairs of points that make up all sides of the polygon.
        self.adj_polygon_points = zip(*[[self.polygon[-1]] + self.polygon[:-1],
                                              self.polygon,
                                              self.polygon[1:] + [self.polygon[0]]])

        # Vectors of coefficients.
        (h1_coefs, h2_coefs, h3_coefs, rhs_coefs) = ([], [], [], [])

        for (prev_point, curr_point, next_point) in self.adj_polygon_points:
            # Choose clockwise order when constructing line of eq. a x + b y + c = 0 from 2 given
            # points. Then the semi-plane containing the convex polygon will be given by:
            # a x + b y + c <= 0.
            line = Line(Point2D(next_point), Point2D(curr_point))
            self.polygon_lines.append(line)

            line_equation_poly = poly(str(line.equation()), gens=sp.symbols(['x', 'y']))

            # Check if a x + b y + c <= 0 holds; if not, flip the line equation -- critical step!
            if line_equation_poly.eval(prev_point) > 0:
                line_equation_poly = -line_equation_poly

            gradient_poly = poly(str(line_equation_poly.eval(gradient)),
                                 gens=sp.symbols(['h0', 'h1', 'h2']))

            h1_coefs.append(float(gradient_poly.diff('h0').eval((0, 0, 0))))
            h2_coefs.append(float(gradient_poly.diff('h1').eval((0, 0, 0))))
            h3_coefs.append(float(gradient_poly.diff('h2').eval((0, 0, 0))))
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

            # Perform triangulation w.r.t to a fixed vertex.
            return self.triangulate(0) or self.triangulate(1) or self.triangulate(2)


    def triangulate(self, vertex):
        others = [0, 1, 2, 0, 1]
        others = others[(vertex + 1):(vertex + 3)]
        segment = Segment(self.triangle[others[0]], self.triangle[others[1]])
        vertex_str = 'h' + str(vertex)
        segment_heights_str = ['h' + str(other) for other in others]

        new_points = []
        counter = 2

        for line in self.polygon_lines:
            perp = line.perpendicular_line(Point2D(self.triangle[vertex][0],
                                                   self.triangle[vertex][1]))
            intersections = perp.intersection(segment)

            if len(intersections) > 0:
                counter = counter + 1
                new_points.append(Point3D(intersections[0].x,
                                          intersections[0].y,
                                          'h' + str(counter)))

        no_vars = counter + 1
        all_side_points = [Point3D(segment.points[0][0],
                                   segment.points[0][1],
                                   segment_heights_str[0])] + new_points + \
                          [Point3D(segment.points[1][0],
                                   segment.points[1][1],
                                   segment_heights_str[1])]
        all_pairs = zip(all_side_points, all_side_points[1:])

        h_coefs = matrix(0.0, (len(all_pairs) * len(self.polygon), no_vars))
        rhs_coefs = []

        for (idx1, pair) in enumerate(all_pairs):
            # Compute gradient for each triangulation.
            gradient = compute_plane_gradient(
                Point3D(self.triangle[vertex][0], self.triangle[vertex][1], vertex_str),
                        pair[0],
                        pair[1])

            heights_str = [str(pair[0][2]), str(pair[1][2])]
            hs = [int(heights_str[0][1:]), int(heights_str[1][1:])]

            # Ensure that each gradient is with the convex polygon.
            for (idx2, (prev_point, curr_point, next_point)) in enumerate(self.adj_polygon_points):
                line = Line(Point2D(next_point), Point2D(curr_point))
                line_equation_poly = poly(str(line.equation()), gens=sp.symbols(['x', 'y']))
                # Flip equation if necessary.
                if line_equation_poly.eval(prev_point) > 0:
                    line_equation_poly = -line_equation_poly

                gradient_poly = poly(str(line_equation_poly.eval(gradient)),
                                     gens=sp.symbols([vertex_str, heights_str[0], heights_str[1]]))

                row = idx1 * len(self.polygon) + idx2
                h_coefs[row, 0] = float(gradient_poly.diff(vertex_str).eval((0, 0, 0)))
                h_coefs[row, hs[0]] = float(gradient_poly.diff(heights_str[0]).eval((0, 0, 0)))
                h_coefs[row, hs[1]] = float(gradient_poly.diff(heights_str[1]).eval((0, 0, 0)))
                rhs_coefs.append(-float(gradient_poly.eval((0, 0, 0))))

        ones = [1.0] * no_vars

        # Final LP Problem.
        # Objective function.
        objective_function_vector = matrix(ones)

        # Coefficient matrix.
        ones_matrix = spmatrix(ones, range(no_vars), range(no_vars))
        heights_coef_matrix = h_coefs
        bounds_coef_matrix = matrix([ones_matrix, -ones_matrix])
        coef_matrix = sparse([heights_coef_matrix, bounds_coef_matrix])

        # Column vector of right hand sides.
        column_vector = matrix(rhs_coefs + \
                               [self.bounds[1]] * no_vars + [-self.bounds[0]] * no_vars)

        min_sol = solvers.lp(objective_function_vector, coef_matrix, column_vector)
        is_consistent = min_sol['x'] is not None

        if is_consistent:
            self.min_heights = np.array(min_sol['x'])
            print np.around(self.min_heights, decimals=2)

            max_sol = solvers.lp(-objective_function_vector, coef_matrix, column_vector)
            self.max_heights = np.array(max_sol['x'])
            print np.around(self.max_heights, decimals=2)

        return is_consistent



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
    cons.plotDomain()


if __name__ == '__main__':
    main()
