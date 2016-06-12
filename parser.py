#/usr/local/bin/python

from utils import Utils

import numpy as np
import re

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'

GRID_INFO_STRING       = r'# Grid information'
POLYNOMIAL_INFO_STRING = r'# Polynomial information'
FUNCTION_INFO_STRING   = r'# Function information'
DERIVATIVE_INFO_STRING = r'# Derivative information'
RANDOM_HEIGHTS_STRING  = r'# Random heights'

FLOAT_NUMBER_REGEX = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
DIMENSION_REGEX    = r'# Dimension: (\d+)'

class Parser:

    @staticmethod
    def init_dimension(input_file):
        with open(input_file, 'r') as f:
            return int(re.search(DIMENSION_REGEX, f.readline()).group(1))


    @staticmethod
    def init_grid_info(input_file, dimension):
        grid_list = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(GRID_INFO_STRING):
                    grid_list = [map(float, next(f).split()) for x in xrange(dimension)]
                    return np.array(grid_list)


    @staticmethod
    def init_no_points_per_axis(grid_info):
        return [len(grid_info[axis]) for axis in range(len(grid_info))]


    @staticmethod
    def init_random_heights(input_file, dimension, no_points_per_axis):
        flat_random_heights = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(RANDOM_HEIGHTS_STRING):
                    for line in f:
                        if line.startswith(FUNCTION_INFO_STRING):
                            break
                        if line.startswith('#') or line.startswith('\n'):
                            continue
                        flat_random_heights.extend(line.rstrip().split(' '))

        return np.array(flat_random_heights, dtype=float).reshape(no_points_per_axis)


    @staticmethod
    def init_function_info(input_file, dimension, no_points_per_axis):
        function_info_lines = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(FUNCTION_INFO_STRING):
                    for line in f:
                        if line.startswith(DERIVATIVE_INFO_STRING):
                            break
                        function_info_lines.append(line.rstrip())

        return Parser.parse_tuples_info(function_info_lines, dimension, no_points_per_axis, True)


    @staticmethod
    def init_derivative_info(input_file, dimension, no_points_per_axis):
        derivative_info_lines = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith(DERIVATIVE_INFO_STRING):
                    for line in f:
                        derivative_info_lines.append(line.rstrip())

        return Parser.parse_tuples_info(derivative_info_lines, dimension, no_points_per_axis, False)


    @staticmethod
    def build_tuples_regex_for_dimension(d):
        return r'\((?P<' + LOWER_BOUND + r'_' + str(d) + r'>' + FLOAT_NUMBER_REGEX + r'),\s*' + \
                 r'(?P<' + UPPER_BOUND + r'_' + str(d) + r'>' + FLOAT_NUMBER_REGEX + r')\)'


    @staticmethod
    def build_tuples_regex(n, is_function_info):
        # Constructs a regex to match either (x, y) - for function information, or
        # ((a, b), (c, d), ...) - for derivative information.
        if is_function_info:
            return Parser.build_tuples_regex_for_dimension(1)

        return r'\(' + r',\s*'.join([Parser.build_tuples_regex_for_dimension(d + 1)
                                     for d in range(n)]) + r'\)'


    @staticmethod
    def build_tuple_match(n, match, is_function_info):
        if is_function_info:
            return tuple([match.group(LOWER_BOUND + '_1'), match.group(UPPER_BOUND + '_1')])

        return tuple([(match.group(LOWER_BOUND + '_%d' % (d + 1)), \
                       match.group(UPPER_BOUND + '_%d' % (d + 1))) for d in range(n)])


    @staticmethod
    def parse_tuples_info(lines, dimension, no_points_per_axis, is_function_info):
        flat_nd_list = []
        regex = Parser.build_tuples_regex(dimension, is_function_info)

        for line in lines:
            # Ignore possible comments in the input lines.
            if line.startswith('#'):
                continue

            # Append the pairs/tuples of lower and upper bounds to the flat list.
            for match in re.finditer(regex, line):
                flat_nd_list.append(Parser.build_tuple_match(dimension, match, is_function_info))

        # Finally, convert to the shape of an n-dimensional array from the given points.
        return np.array(flat_nd_list, \
                        dtype=Utils.get_dtype(dimension, is_function_info)).reshape(
                        no_points_per_axis)

