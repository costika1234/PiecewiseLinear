#/usr/local/bin/python

from sympy import poly
from utils import Utils
from parser import Parser

import numpy as np
import sympy as sp

POLY_TERMS = 10
POLY_DEGREE = 10
POLY_MIN_COEF = -40
POLY_MAX_COEF = 40

RANDOM_LOWER_BOUND = 10
RANDOM_UPPER_BOUND = 20

class InputGenerator:
    
    def __init__(self, input_file, n, no_points_per_axis, from_poly=False, eps=20.0):
        self.input_file = input_file
        self.n = n
        self.no_points_per_axis = [int(i) for i in no_points_per_axis.split(' ')]
        self.from_poly = from_poly
        self.random_heights = np.zeros(self.no_points_per_axis)
        self.no_vars = np.prod(self.no_points_per_axis)
        self.grid_info = None 
        self.eps = eps


    def init_random_heights(self):
        flat_random_heights = np.empty(self.no_vars, dtype=float)

        if self.from_poly:
            poly = self.build_random_polynomial()
            grid_indices = Utils.get_grid_indices(self.no_points_per_axis, ignore_last=False)

            for index, grid_index in enumerate(grid_indices):
                coord = Utils.convert_grid_index_to_coord(grid_index, self.grid_info)
                flat_random_heights[index] = poly.eval(coord)
        else:
            flat_random_heights = (RANDOM_UPPER_BOUND - RANDOM_LOWER_BOUND) * \
                                   np.random.random_sample((self.no_vars,)) + RANDOM_LOWER_BOUND

        self.random_heights = flat_random_heights.reshape(self.no_points_per_axis)


    def build_random_polynomial(self):
        # Construct list of random coefficients and exponents. 
        coef_exps_list = []
        for index in range(POLY_TERMS):
            coef = np.random.randint(POLY_MIN_COEF, POLY_MAX_COEF, 1).tolist()
            exps = np.random.randint(0, POLY_DEGREE, self.n).tolist()
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
        flat_info = []
        grid_indices = Utils.get_grid_indices(self.no_points_per_axis, ignore_last=False)

        for grid_index in grid_indices:
            coord = Utils.convert_grid_index_to_coord(grid_index, self.grid_info)
            f_value = self.random_heights[grid_index]

            if is_function_info:
                # Ensure that we generate wide enough intervals that contain all adjacent points. 
                if not Utils.is_border_index(grid_index, self.no_points_per_axis):
                    grid_indices_neighbours = Utils.get_grid_indices_neighbours(grid_index)
                    min_h, max_h = float('inf'), float('-inf')

                    for grid_index_neighbour in grid_indices_neighbours:
                        f_value_neighbour = self.random_heights[grid_index_neighbour]
                        min_h, max_h = min(min_h, f_value_neighbour), max(max_h, f_value_neighbour)

                    f_interval = (min_h - self.eps, max_h + self.eps)
                else:
                    # For border points (i.e. those who have at least one coordinate on the
                    # rightmost endpoint of each axis), initialize a default interval, as the values
                    # will never be used.
                    f_interval = (0.0, 0.0)

                flat_info.append(f_interval)

            else:
                d_values = [0.0] * self.n
                # Construct intervals for each partial derivative.
                for axis in range(self.n):
                    grid_index_neighbour = Utils.get_grid_index_neighbour_for_axis(grid_index, axis)
                    # Ensure that we do not step outside the grid when deriving 'b_ij' values 
                    # according to: b_ij = (h_i+1,j - h_ij) / (p_i+1 - p_i).
                    if grid_index_neighbour[axis] < self.no_points_per_axis[axis]:
                        f_value_neighbour = self.random_heights[grid_index_neighbour]
                        coord_neighbour = Utils.convert_grid_index_to_coord(grid_index_neighbour,
                                                                            self.grid_info)            
                        d_values[axis] = (f_value_neighbour - f_value) / \
                                         (coord_neighbour[axis] - coord[axis])
                
                # Convert the derived values to intervals (b-, b+).
                d_intervals = tuple([(d_val - self.eps, d_val + self.eps) for d_val in d_values])
                flat_info.append(d_intervals)

        return flat_info


    def generate_tuples_info(self, file_descriptor, is_function_info):
        dt = Utils.get_dtype(self.n, is_function_info)

        flat_info = self.generate_flat_info(is_function_info)
        nd_array = np.array(flat_info, dtype=dt).reshape(self.no_points_per_axis)

        # Write contents to file.
        file_descriptor.write('# Array shape: {0}\n'.format(nd_array.shape))
        self.traverse_nd_array(nd_array, file_descriptor, self.n)


    def generate_test_file(self):
        with open(self.input_file, 'w+') as f:
            # Dimension.
            f.write('# Dimension: %d\n\n' % self.n)

            # Grid points (equally spaced on each of the 'n' axis of the domain).
            f.write('# Grid information (each of the %d lines specify divisions on the domain axis,'
                    ' in strictly increasing order. The endpoints will therefore specify the '
                    'constraints for the function domain):\n' % self.n)
            for no_points in self.no_points_per_axis:
                np.savetxt(f, np.linspace(0.0, 1.0, no_points), newline=' ', fmt='%s')
                f.write('\n')

        # Initialize random values for heights (either from polynomial, or randomly chosen).
        self.grid_info = Parser.init_grid_info(self.input_file, self.n)
        self.init_random_heights()

        with open(self.input_file, 'a+') as f:
            # Function information.
            f.write('\n# Function information (specified as a %d-dimensional array of intervals, '
                    'where an entry is of the form (c-, c+), c- <= c+, and represents the '
                    'constraint for the function value at a particular grid point):\n' % self.n)
            self.generate_tuples_info(f, is_function_info=True)

            # Derivative information.
            f.write('\n# Derivative information (specified as a %d-dimensional array of tuples of '
                    'intervals, where an entry is of the form ((c1-, c1+), (c2-, c2+), ...), '
                    'ci- <= ci+, and represents the constraints along each partial derivative at a '
                    'particular grid point):\n' % self.n)
            self.generate_tuples_info(f, is_function_info=False)

