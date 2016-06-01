#/usr/local/bin/python

from cvxopt import spmatrix
from operator import mul

import contextlib
import itertools
import numpy as np
import sys

class DummyFile(object):
    def write(self, x):
        pass

class Utils:

    @staticmethod
    @contextlib.contextmanager
    def nostdout():
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout


    @staticmethod
    def get_grid_indices(no_points_per_axis, ignore_last):
        offset = 0
        if ignore_last:
            offset = -1

        return list(itertools.product(*[range(no_pts + offset) for no_pts in no_points_per_axis]))


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


    @ staticmethod
    def get_dimension_for_row_vector(no_points_per_axis, axis):
        result = list(no_points_per_axis)
        result[axis] = result[axis] - 1
        return result


    @staticmethod
    def get_grid_index_neighbour_for_axis(grid_index, axis):
        result = list(grid_index)
        result[axis] = result[axis] + 1
        return tuple(result)


    @staticmethod
    def get_grid_indices_excluding_last_point_on_axis(no_points_per_axis, axis):
        args = []
        for index, no_points in enumerate(no_points_per_axis):
            if index == axis:
                args.append(range(no_points - 1))
            else:
                args.append(range(no_points))

        return list(itertools.product(*args))


    @staticmethod
    def get_partial_derivatives_end_points(dimension):
        no_cols = 2**(dimension - 1)
        result = np.zeros((dimension, no_cols), dtype=('int, int'))

        for index, row in enumerate(result):
            step = 2**(dimension - index - 1)
            for j in range(step):
                row[j][0] = j
                row[j][1] = row[j][0] + step

        for index in range(1, len(result)):
            step = 2**(dimension - index - 1)
            offset = 2 * step

            for j in range(no_cols):
                if j + step < no_cols:
                    result[index][j + step][0] = result[index][j][0] + offset
                    result[index][j + step][1] = result[index][j][1] + offset

        return result


    @staticmethod
    def get_dtype(dimension, is_function_info):
        if is_function_info:
            return ('float64, float64')

        tuple_dtype = [('lower_bound', 'float64'), ('upper_bound', 'float64')]
        return [(str(dimension + 1), tuple_dtype) for dimension in range(dimension)]


    @staticmethod
    def calculate_block_heights(no_points_per_axis):
        dimensions_prod = reduce(mul, no_points_per_axis)
        return [dimensions_prod / no_points_per_axis[i] * (no_points_per_axis[i] - 1) \
                for i in range(len(no_points_per_axis))]


    @staticmethod
    def calculate_number_of_sub_blocks(ith_partial_derivative, no_points_per_axis):
        if ith_partial_derivative == 0:
            return 1

        return reduce(mul, no_points_per_axis[:ith_partial_derivative])


    @staticmethod
    def calculate_distance_between_non_zero_entries(ith_partial_derivative, no_points_per_axis):
        if ith_partial_derivative == len(no_points_per_axis) - 1:
            return 1

        return reduce(mul, no_points_per_axis[(ith_partial_derivative + 1):])


    @staticmethod
    def calculate_adjacent_sub_block_offsets(width, no_points_per_axis):
        block_heights = Utils.calculate_block_heights(no_points_per_axis)

        result = []
        for index, no_points in enumerate(no_points_per_axis):
            no_sub_blocks = Utils.calculate_number_of_sub_blocks(index, no_points_per_axis)
            if no_sub_blocks == 1:
                result += [0]
            else:
                sub_block_width = width / no_sub_blocks
                sub_block_height = block_heights[index] / no_sub_blocks
                result += [sub_block_width - sub_block_height]

        return result


    @staticmethod
    def build_matrix_for_partial_derivative(offset, block_height, no_sub_blocks, distance):
        # 'block_height' specifies the height of the current block matrix.
        # 'no_sub_blocks' specifies the number of subblocks in the current block matrix.
        # 'distance' specifies the distance between the -1 and 1 entries for this block matrix.
        # TODO: Refactor this method heavily!

        minus_ones_and_ones = [-1] * block_height + [1] * block_height
        row_range = range(block_height) * 2

        if no_sub_blocks == 1:
            column_range_minus_ones = range(block_height)
        else:
            column_range_minus_ones = []

            n = 0
            for index in range(block_height):
                if index % (block_height / no_sub_blocks) == 0 and index != 0:
                    n = n + offset

                column_range_minus_ones += [n]
                n = n + 1

        column_range = column_range_minus_ones + [x + distance for x in column_range_minus_ones]

        return spmatrix(minus_ones_and_ones, row_range, column_range)


    @staticmethod
    def get_triangulation(x_dim, y_dim):
        triangles = []

        for i in range(x_dim - 1):
            for j in range(y_dim - 1):
                label = y_dim * i + j
                triangles.append([label, label + y_dim, label + 1])
                triangles.append([label + y_dim, label + y_dim + 1, label + 1])

        return triangles


    @staticmethod
    def generate_random_points_per_axis(no_points):
        random_points_list = [0.0]

        if no_points > 2:
            eq_spaced_points_list = np.linspace(0.0, 1.0, no_points)
            offset = 1.0 / (4 * (no_points - 1))

            for index in range(no_points - 2):
                point = eq_spaced_points_list[index + 1]
                random_points_list.append(np.random.uniform(point - offset, point + offset))

        random_points_list.append(1.0)

        return random_points_list

