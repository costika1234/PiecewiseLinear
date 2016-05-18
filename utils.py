#/usr/local/bin/python
 
from cvxopt import matrix, solvers, sparse, spmatrix
from mpl_toolkits.mplot3d import Axes3D
from operator import mul
from optparse import OptionParser
from sympy import poly

import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
import sympy as sp
import sys

class Utils:

    @staticmethod
    def get_grid_indices(no_points_per_axis, ignore_last):
        offset = 0
        if ignore_last:
            offset = -1

        return list(itertools.product(*[range(no_points + offset) \
                    for no_points in no_points_per_axis]))


    @staticmethod
    def convert_grid_index_to_coord(grid_index, grid_info):
        return tuple([grid_info[i][grid_index[i]] for i in range(len(grid_info))])


    @staticmethod
    def is_border_index(grid_index, no_points_per_axis):
        for i in range(len(no_points_per_axis)):
            if grid_index[i] == no_points_per_axis[i] - 1:
                return True

        return False


    @staticmethod
    def get_grid_indices_neighbours(grid_index):
        tuple_sum_lambda = lambda x, y: tuple(map(sum, zip(x, y)))
        all_binary_perms = list(itertools.product([0, 1], repeat=len(grid_index)))

        return [tuple_sum_lambda(grid_index, perm) for perm in all_binary_perms]


    @staticmethod
    def get_grid_index_neighbour_for_axis(grid_index, axis):
        result = list(grid_index)
        result[axis] = result[axis] + 1
        return tuple(result)