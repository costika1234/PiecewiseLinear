#/usr/local/bin/python

from parser import Parser
from sympy import poly
from utils import Utils

import copy_reg
import numpy as np
import sympy as sp
import multiprocessing
import types

POLY_TERMS = 10
POLY_MAX_DEGREE = 10
POLY_MIN_COEF = -40
POLY_MAX_COEF = 40
OFFSET = 0.1

RANDOM_LOWER_BOUND = 10
RANDOM_UPPER_BOUND = 20

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class InputGenerator:

    def __init__(self, input_file, is_cons_input, n, no_points_per_axis, rand_points_axis=False,
                 from_poly=False, eps=0.0):
        # Flag which specifies whether consistent input will be generated or not.
        self.is_cons_input = is_cons_input

        # Path to file in which data will be generated.
        self.input_file = input_file

        # Domain dimension (integer).
        self.n = n

        # Number of points on each of the 'n' axis (list).
        self.no_points_per_axis = [int(i) for i in no_points_per_axis.split(' ')]

        # Specifies whether the points per each axis are generated randomly or not.
        self.rand_points_axis = rand_points_axis

        # Randomly generated heights to enable automatic testing (n-dim numpy array).
        self.random_heights = np.zeros(self.no_points_per_axis).reshape(self.no_points_per_axis)

        # Grid information (numpy array).
        self.grid_info = None

        # Function information (n-dim numpy array of pairs).
        self.function_info = None

        # Derivative information (n-dim numpy array of tuples).
        self.derivative_info = None

        # Number of decision variables (integer).
        self.no_vars = np.prod(self.no_points_per_axis)

        # Tolerance for generating random intervals (float).
        self.eps = eps

        # Flag which specifies whether random heights are generated from a polynomial or not.
        self.from_poly = from_poly


    def init_random_heights(self):
        if self.from_poly:
            poly = self.build_random_polynomial()
            grid_indices = Utils.get_grid_indices(self.no_points_per_axis, ignore_last=False)

            # Parallelise the initialisation of random values derived from the polynomial.
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            coords = [Utils.convert_grid_index_to_coord(grid_index, self.grid_info)
                      for grid_index in grid_indices]

            self.random_heights = np.array(pool.map(poly.eval, coords),
                                           dtype=float).reshape(self.no_points_per_axis)
            pool.close()
            pool.join()
        else:
            self.random_heights = np.random.uniform(RANDOM_LOWER_BOUND,
                                                    RANDOM_UPPER_BOUND,
                                                    self.no_points_per_axis)


    def build_random_polynomial(self):
        # Construct list of random coefficients and exponents.
        coef_exps_list = []
        for index in range(POLY_TERMS):
            coef = np.random.randint(POLY_MIN_COEF, POLY_MAX_COEF, 1).tolist()
            exps = np.random.randint(0, POLY_MAX_DEGREE, self.n).tolist()
            coef_exps_list.append(tuple(coef + exps))

        # Construct polynomial variables as x_1, x_2, ..., x_n.
        poly_vars = sp.symbols(['x_' + str(i + 1) for i in range(self.n)])

        # Construct the terms of the polynomial.
        poly_terms = []
        for coef_exps in coef_exps_list:
            coef = coef_exps[0]
            poly_term = []
            for index, exponent in enumerate(coef_exps[1:]):
                poly_term.append(str(poly_vars[index]) + '**' + str(exponent))

            poly_terms.append(str(coef) + ' * ' + ' * '.join(poly_term))

        return poly(' + '.join(poly_terms), gens=poly_vars)


    def traverse_nd_array(self, nd_array, f, depth):
        # Function which recursively traverses an n-dimenionsal array and saves the array to file.
        if depth == 2:
            np.savetxt(f, nd_array, fmt='%s')
        else:
            for sub_nd_array in nd_array:
                self.traverse_nd_array(sub_nd_array, f, depth - 1)

        f.write('# End of %d depth \n' % depth)


    def generate_flat_info(self, is_function_info):
        # Reverse engineer the LP algorithm to construct either consistent or inconsistent input.
        flat_info = []
        grid_indices = Utils.get_grid_indices(self.no_points_per_axis, ignore_last=False)

        # In order to generate inconsistent input, change a random partial derivative at a random
        # grid point so that the derivative is always strictly greater than the gradient within the
        # corresponding hyper-rectangle. This is the minimal change we can make to get inconsistent
        # input.
        if not self.is_cons_input and self.function_info is not None:
            interior_grid_indices = Utils.get_grid_indices(self.no_points_per_axis,
                                                           ignore_last=True)
            random_index_in_list = np.random.randint(0, len(interior_grid_indices))
            random_interior_grid_index = interior_grid_indices[random_index_in_list]
            random_partial_derivative = np.random.randint(0, self.n)

        for grid_index in grid_indices:
            if is_function_info:
                f_interval = (0.0, 0.0)
                f_value = self.random_heights[grid_index]
                # Ensure that we generate wide enough intervals that contain all adjacent points.
                if not Utils.is_border_index(grid_index, self.no_points_per_axis):
                    grid_indices_neighbours = Utils.get_grid_indices_neighbours(grid_index)
                    min_h, max_h = float('inf'), float('-inf')

                    for next_grid_index in grid_indices_neighbours:
                        f_value_neighbour = self.random_heights[next_grid_index]
                        min_h, max_h = min(min_h, f_value_neighbour), max(max_h, f_value_neighbour)

                    if self.is_cons_input:
                        # To guarantee consistency, the interval for function information at this
                        # particulat grid point must include at least [min_h, max_h].
                        # Use the epsilon to vary this interval and to allow a greater distance
                        # between the minimal and maximal surfaces.
                        f_interval = (min_h - self.eps, max_h + self.eps)
                    else:
                        # For inconsistent input, simply take the least box containing the generated
                        # heights.
                        f_interval = (min_h, max_h)

                flat_info.append(f_interval)

            else:
                d_values = [(0.0, 0.0)] * self.n
                if not Utils.is_border_index(grid_index, self.no_points_per_axis):
                    # Along each partial derivative and for all possible triangulations, compute
                    # the minimum and maximum gradients based on the given heights.
                    for ith_partial in range(self.n):
                        partial_derivatives_end_points = Utils.get_partial_derivatives_end_points(
                            self.n)
                        next_grid_indices = Utils.get_grid_indices_neighbours(grid_index)
                        min_b, max_b = float('inf'), float('-inf')

                        for end_points in partial_derivatives_end_points[ith_partial]:
                            next_index = next_grid_indices[end_points[1]]
                            curr_index = next_grid_indices[end_points[0]]

                            next_coord = Utils.convert_grid_index_to_coord(next_index,
                                self.grid_info)
                            curr_coord = Utils.convert_grid_index_to_coord(curr_index,
                                self.grid_info)

                            f_value_next = self.random_heights[next_index]
                            f_value_curr = self.random_heights[curr_index]
                            grid_diff = next_coord[ith_partial] - curr_coord[ith_partial]
                            gradient = (f_value_next - f_value_curr) / grid_diff

                            min_b, max_b = min(min_b, gradient), max(max_b, gradient)

                        if self.is_cons_input:
                            d_values[ith_partial] = (min_b - self.eps, max_b + self.eps)
                        else:
                            # Make the minimal change to guarantee inconsistent input.
                            if random_interior_grid_index == grid_index and \
                               random_partial_derivative == ith_partial:
                                f_lower = self.function_info[random_interior_grid_index][0]
                                f_upper = self.function_info[random_interior_grid_index][1]
                                max_gradient = abs(f_upper - f_lower) / grid_diff

                                inconsistent_interval = (max_gradient + OFFSET, 3 * max_gradient)
                                d_values[ith_partial] = inconsistent_interval
                            else:
                                d_values[ith_partial] = (min_b, max_b)

                flat_info.append(tuple(d_values))

        return flat_info


    def generate_tuples_info(self, file_descriptor, is_function_info):
        dt = Utils.get_dtype(self.n, is_function_info)

        flat_info = self.generate_flat_info(is_function_info)
        nd_array = np.array(flat_info, dtype=dt).reshape(self.no_points_per_axis)

        # Write contents to file.
        file_descriptor.write('# Array shape: {0}\n'.format(nd_array.shape))
        self.traverse_nd_array(nd_array, file_descriptor, self.n)

        return nd_array


    def generate_test_file(self):
        with open(self.input_file, 'w+') as f:
            # Dimension.
            f.write('# Dimension: %d\n\n' % self.n)

            # Grid points (equally spaced on each of the 'n' axis of the domain).
            f.write('# Grid information (each of the %d lines specify divisions on the domain axis,'
                    ' in strictly increasing order. The endpoints will therefore specify the '
                    'constraints for the function domain):\n' % self.n)

            for no_points in self.no_points_per_axis:
                if self.rand_points_axis:
                    points_list = Utils.generate_random_points_per_axis(no_points)
                else:
                    points_list = np.linspace(0.0, 1.0, no_points)

                np.savetxt(f, points_list, newline=' ', fmt='%s')
                f.write('\n')

        # Initialize random values for heights (either from polynomial, or randomly chosen).
        self.grid_info = Parser.init_grid_info(self.input_file, self.n)
        self.init_random_heights()

        with open(self.input_file, 'a+') as f:
            # Function information.
            f.write('\n# Function information (specified as a %d-dimensional array of intervals, '
                    'where an entry is of the form (c-, c+), c- <= c+, and represents the '
                    'constraint for the function value at a particular grid point):\n' % self.n)
            self.function_info = self.generate_tuples_info(f, is_function_info=True)

            # Derivative information.
            f.write('\n# Derivative information (specified as a %d-dimensional array of tuples of '
                    'intervals, where an entry is of the form ((c1-, c1+), (c2-, c2+), ...), '
                    'ci- <= ci+, and represents the constraints along each partial derivative at a '
                    'particular grid point):\n' % self.n)
            self.derivative_info = self.generate_tuples_info(f, is_function_info=False)

