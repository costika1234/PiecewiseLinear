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


LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'

GRID_INFO_STRING       = r'# Grid information'
FUNCTION_INFO_STRING   = r'# Function information'
DERIVATIVE_INFO_STRING = r'# Derivative information'

#FLOAT_NUMBER_REGEX = r'[-+]?[0-9]*\.?[0-9]+'
FLOAT_NUMBER_REGEX = r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
DIMENSION_REGEX    = r'# Dimension: (\d+)'

####################################################################################################
######################################### DATA GENERATION ##########################################
####################################################################################################

def get_polynomial_vars(n):
    return sp.symbols(['x_' + str(i + 1)for i in range(n)])


def get_coefs_exponents_list(n):
    result = []
    for index in range(20):
        coef = np.random.randint(-20, 20, 1).tolist()
        result.append(tuple([coef[0]] + np.random.randint(0, 10, n).tolist()))

    print result
    return result


def build_polynomial(coefs_exponents_list):
    # Construct an 'n'-dimensional polynomial from the given coefficients and exponents list, 
    # given in the form: [(coef_1, exp_1, exp_2, ..., exp_n), ...].
    poly_terms = []
    poly_vars = get_polynomial_vars(len(coefs_exponents_list[0][1:]))

    for coef_exponents in coefs_exponents_list:
        coef = coef_exponents[0]
        poly_term = []
        for index, exponent in enumerate(coef_exponents[1:]):
            poly_term.append(str(poly_vars[index]) + '**' + str(exponent))
        
        poly_terms.append(str(coef) + ' * ' + " * ".join(poly_term))

    return poly(' + '.join(poly_terms), gens=poly_vars)


def get_flat_info_from_polynomial(polynomial, grid_info, no_points_per_axis, is_function_info):
    n = len(grid_info)

    if not is_function_info:
        poly_vars_str = [str(var) for var in get_polynomial_vars(n)]
        derivatives = [polynomial.diff(poly_vars_str[i]) for i in range(n)]
        
    grid_points = generate_indices(no_points_per_axis, False)
    flat_info = []
    eps = 10.0

    # Generate intervals from the given polynomial. 
    for grid_point in grid_points:
        point = tuple([grid_info[i][grid_point[i]] for i in range(n)])
        
        if is_function_info:
            f_value = float(polynomial.eval(point))
            f_interval = (f_value - eps, f_value + eps)
            flat_info.append(f_interval)
        else:
            d_values = [float(derivative.eval(point)) for derivative in derivatives]
            d_intervals = tuple([(d_value - 3 * eps, d_value + 3 * eps) for d_value in d_values])
            flat_info.append(d_intervals)

    return flat_info


def get_zipped_list(no_elements):
    # Returns a list of the form: [(a, b), ...] to help generating dummy data.
    # l = np.array([round(x, 2) for x in np.linspace(no_elements, 1.0, no_elements)])
    l = np.random.randint(10, 20, no_elements)
    # u = l + no_elements
    u = l + 10
    return zip(l, u)


def get_zipped_list_2(no_elements):
    # Returns a list of the form: [(a, b), ...] to help generating dummy data.
    l = np.random.randint(-20, 0, no_elements)
    u = l + 20
    return zip(l, u)


def get_dtype(dimension, is_function_info):
    if is_function_info:
        return ('float64, float64')

    tuple_dtype = [('lower_bound', 'float64'), ('upper_bound', 'float64')]
    return [(str(dimension + 1), tuple_dtype) for dimension in range(dimension)]


def traverse_nd_array(nd_array, f, depth):
    # Function which recursively traverses an n-dimenionsal array and saves the array to file.
    if depth == 2:
        np.savetxt(f, nd_array, fmt='%s')
    else:
        for sub_nd_array in nd_array:
            traverse_nd_array(sub_nd_array, f, depth - 1)

    f.write('# End of %d depth \n' % depth)


def generate_tuples_info(file_descriptor, n, no_points_per_axis, grid_info, is_function_info, 
                         polynomial):
    no_elements = np.prod(no_points_per_axis)
    dt = get_dtype(n, is_function_info)

    if is_function_info:
        # Create flat array with pairs (c-, c+).
        if polynomial:
            flat_function_info = get_flat_info_from_polynomial(polynomial, 
                                                               grid_info, 
                                                               no_points_per_axis, 
                                                               is_function_info) 
        else:
            flat_function_info = get_zipped_list(no_elements)

        nd_array = np.array(flat_function_info, dtype=dt).reshape(no_points_per_axis)
    else:
        # Create flat array with tuples ((c1-, c1+), (c2-, c2+), ...).
        if polynomial:
            flat_derivative_info = get_flat_info_from_polynomial(polynomial, 
                                                                 grid_info, 
                                                                 no_points_per_axis, 
                                                                 is_function_info) 
        else:
            zipped = get_zipped_list_2(no_elements)
            flat_derivative_info = zip(*[zipped for _ in range(n)])
            
        nd_array = np.array(flat_derivative_info, dtype=dt).reshape(no_points_per_axis)

    # Write contents to file.
    file_descriptor.write('# Array shape: {0}\n'.format(nd_array.shape))
    traverse_nd_array(nd_array, file_descriptor, n)


def generate_test_file(input_file, n, no_points_per_axis, from_poly):
    # Initialize the number of points on each axis that will determine the grid.
    # no_points_per_axis = tuple([x + 20 for x in range(n)])
    no_points_per_axis = tuple([no_points_per_axis] * n)
    if from_poly:
        polynomial = build_polynomial(get_coefs_exponents_list(n))

    with open(input_file, 'w+') as f:
        # Dimension.
        f.write('# Dimension: %d\n\n' % n)

        # Grid points.
        f.write('# Grid information (each of the %d lines specify divisions on the domain axis, in '
                'strictly increasing order. The endpoints will therefore specify the constraints '
                'for the function domain):\n' % n)
        for no_points in no_points_per_axis:
            np.savetxt(f, np.linspace(0.0, 1.0, no_points), newline=' ', fmt='%s')
            f.write('\n')

    # Retrieve the grid points.
    grid_info = init_grid_info(input_file, n)
        
    with open(input_file, 'a+') as f:
        # Function information.
        f.write('\n# Function information (specified as a %d-dimensional array of intervals, where '
                'an entry is of the form (c-, c+), c- < c+, and represents the constraint for the '
                'function value at a particular grid point):\n' % n)
        generate_tuples_info(f, n, no_points_per_axis, grid_info, True, polynomial)

        # Derivative information.
        f.write('\n# Derivative information (specified as a %d-dimensional array of tuples of '
                'intervals, where an entry is of the form ((c1-, c1+), (c2-, c2+), ...), ci- < ci+,'
                ' and represents the constraints along each partial derivative at a '
                'particular grid point):\n' % n)
        generate_tuples_info(f, n, no_points_per_axis, grid_info, False, polynomial)

####################################################################################################
############################################# PARSING ##############################################
####################################################################################################

def init_dimension(input_file):
    with open(input_file, 'r') as f:
        return int(re.search(DIMENSION_REGEX, f.readline()).group(1))


def init_grid_info(input_file, dimension):
    grid_list = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(GRID_INFO_STRING):
                grid_list = [map(float, next(f).split()) for x in xrange(dimension)]
                return np.array(grid_list)


def init_no_points_per_axis(grid_info):
    return [len(grid_info[axis]) for axis in range(len(grid_info))]


def init_function_info(input_file, dimension, no_points_per_axis):
    function_info_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(FUNCTION_INFO_STRING):
                for line in f:
                    if line.startswith(DERIVATIVE_INFO_STRING):
                        break
                    function_info_lines.append(line.rstrip())

    return parse_tuples_info(function_info_lines, dimension, no_points_per_axis, True)


def init_derivative_info(input_file, dimension, no_points_per_axis):
    derivative_info_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.startswith(DERIVATIVE_INFO_STRING):
                for line in f:
                    derivative_info_lines.append(line.rstrip())

    return parse_tuples_info(derivative_info_lines, dimension, no_points_per_axis, False)


def build_tuples_regex_for_dimension(d):
    return r"\((?P<" + LOWER_BOUND + r"_" + str(d) + r">" + FLOAT_NUMBER_REGEX + r"),\s*" + \
             r"(?P<" + UPPER_BOUND + r"_" + str(d) + r">" + FLOAT_NUMBER_REGEX + r")\)" 


def build_tuples_regex(n, is_function_info):
    # Constructs a regex to match either (x, y) - for function information, or
    # ((a, b), (c, d), ...) - for derivative information.
    if is_function_info:
        return build_tuples_regex_for_dimension(1)

    return r"\(" + r",\s*".join([build_tuples_regex_for_dimension(d + 1) for d in range(n)]) + r"\)"


def build_tuple_match(n, match, is_function_info):
    if is_function_info:
        return tuple([match.group(LOWER_BOUND + "_1"), match.group(UPPER_BOUND + "_1")])

    return tuple([(match.group(LOWER_BOUND + "_%d" % (d + 1)), \
                   match.group(UPPER_BOUND + "_%d" % (d + 1))) for d in range(n)])


def parse_tuples_info(lines, dimension, no_points_per_axis, is_function_info):
    flat_nd_list = []
    regex = build_tuples_regex(dimension, is_function_info)

    for line in lines:
        # Ignore possible comments in the input lines.
        if line.startswith('#'):
            continue

        # Append the pairs/tuples of lower and upper bounds to the flat list.
        for match in re.finditer(regex, line):
            flat_nd_list.append(build_tuple_match(dimension, match, is_function_info))

    # Finally, convert to the shape of an n-dimensional array from the given points.
    return np.array(flat_nd_list, \
                    dtype=get_dtype(dimension, is_function_info)).reshape(no_points_per_axis)

####################################################################################################
################################## LINEAR PROGRAMMING ALGORITHM ####################################
####################################################################################################

def generate_indices(no_points_per_axis, ignore_last_point_on_each_axis):
    offset = 0
    if ignore_last_point_on_each_axis:
        offset = -1

    return list(itertools.product(*[range(no_points + offset) for no_points in no_points_per_axis]))


def generate_indices_ignoring_last_coordinate_on_axis(no_points_per_axis, axis):
    args = []
    for index, no_points in enumerate(no_points_per_axis):
        if index == axis:
            args.append(range(no_points - 1))
        else:
            args.append(range(no_points))

    return list(itertools.product(*args))


def get_shape_ignoring_last_point_on_axis(no_points_per_axis, axis):
    # Need to create a copy so that we do not overwrite values passed by reference.
    result = list(no_points_per_axis)
    result[axis] = result[axis] - 1
    return result


def generate_indices_without_neighbours(no_points_per_axis):
    # TODO: optimise this method by generating the required tuples instead of resorting to
    #       difference of sets.
    indices = generate_indices(no_points_per_axis, False)
    indices_allowing_neighbours = generate_indices(no_points_per_axis, True)

    return set(indices) - set(indices_allowing_neighbours) 


def generate_grid_indices_neighbours(point, n):
    # Given a 'point' in the grid (in terms of indices), determine all its neighbours, along each
    # dimension.
    assert len(point) == n, "Point is not %d dimensional" % n

    tuple_sum_lambda = lambda x, y: tuple(map(sum, zip(x, y)))
    all_binary_perms = list(itertools.product([0, 1], repeat=n))

    return [tuple_sum_lambda(point, perm) for perm in all_binary_perms]


def calculate_block_heights(no_points_per_axis):
    dimensions_prod = reduce(mul, no_points_per_axis)
    return [dimensions_prod / no_points_per_axis[i] * (no_points_per_axis[i] - 1) \
            for i in range(len(no_points_per_axis))]


def calculate_number_of_sub_blocks(ith_partial_derivative, no_points_per_axis):
    if ith_partial_derivative == 0:
        return 1

    return reduce(mul, no_points_per_axis[:ith_partial_derivative])


def calculate_distance_between_non_zero_entries(ith_partial_derivative, no_points_per_axis):
    if ith_partial_derivative == len(no_points_per_axis) - 1:
        return 1

    return reduce(mul, no_points_per_axis[(ith_partial_derivative + 1):])


def calculate_adjacent_sub_block_offsets(width, no_points_per_axis):
    block_heights = calculate_block_heights(no_points_per_axis)

    result = []
    for index, no_points in enumerate(no_points_per_axis):
        no_sub_blocks = calculate_number_of_sub_blocks(index, no_points_per_axis)
        if no_sub_blocks == 1:
            result += [0]
        else:
            sub_block_width = width / no_sub_blocks
            sub_block_height = block_heights[index] / no_sub_blocks
            result += [sub_block_width - sub_block_height]

    return result


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


class Consistency:

    def __init__(self, input_file):
        # Domain dimension (integer).
        self.n = init_dimension(input_file)

        # Grid information (numpy array).
        self.grid_info = init_grid_info(input_file, self.n)

        # Number of points on each of the 'n' axis (list).
        self.no_points_per_axis = init_no_points_per_axis(self.grid_info)

        # Function information (n-dim numpy array of pairs).
        self.function_info = init_function_info(input_file, self.n, self.no_points_per_axis)
        
        # Derivative information (n-dim numpy array of tuples).
        self.derivative_info = init_derivative_info(input_file, self.n, self.no_points_per_axis)

        # Minimum heights for least witness of consistency (n-dim numpy array).
        self.min_heights = np.zeros(self.no_points_per_axis)

        # Maximum heights for greatest witness of consistency (n-dim numpy array).
        self.max_heights = np.zeros(self.no_points_per_axis)

        # Number of decision variables (integer).
        self.no_vars = np.prod(self.no_points_per_axis)


    def solve_LP_problem(self):
        (f_coef_matrix, f_column_vector) = self.build_function_coef_matrix_and_column_vector()
        (d_coef_matrix, d_column_vector) = self.build_derivative_coef_matrix_and_column_vector()        
        
        # Solve the LP problem by combining constraints for both function and derivative info.
        objective_function_vector = matrix(list(itertools.repeat(1.0, self.no_vars)))
        coef_matrix = matrix([f_coef_matrix, d_coef_matrix])
        column_vector = matrix([f_column_vector, d_column_vector])
        
        min_sol = solvers.lp(objective_function_vector, coef_matrix, column_vector)

        # Print the LP problem for debugging purposes.
        self.display_LP_problem(coef_matrix, column_vector)

        if min_sol['x'] is not None:
            self.min_heights = np.array(min_sol['x']).reshape(self.no_points_per_axis)
            print np.around(self.min_heights, decimals=2)

            # Since consistency has been established, solve the converse LP problem to get the
            # maximal bounding surface.
            max_sol = solvers.lp(-objective_function_vector, coef_matrix, column_vector)
            self.max_heights = np.array(max_sol['x']).reshape(self.no_points_per_axis)
            print np.around(self.max_heights, decimals=2)

        else:
            print "No witness for consistency found."


    def build_constraints_from_function_info(self):
        # c- <= h <= c+ and optimize to get the greatest lower bound and least upper bound for the
        # height at any point in the grid. For the contour of the domain for which the previous
        # inequality is not defined, use the function information.
        l_b_constraints = np.empty(self.no_points_per_axis)
        u_b_constraints = np.empty(self.no_points_per_axis)
        
        l_b_constraints.fill(np.float64('-inf'))
        u_b_constraints.fill(np.float64('inf'))

        # For all the grid points that can have a neighbour, optimise the lower and upper bound
        # constraints.
        indices_allowing_neighbours = generate_indices(self.no_points_per_axis, True)
        
        for index in indices_allowing_neighbours:
            neighbour_indices = generate_grid_indices_neighbours(index, self.n)

            for neighbour_index in neighbour_indices:
                l_b_function_value = self.function_info[neighbour_index][0]
                u_b_function_value = self.function_info[neighbour_index][1]

                l_b_constraints[index] = max(l_b_constraints[index], l_b_function_value)
                u_b_constraints[index] = min(u_b_constraints[index], u_b_function_value)

        # For the border indices (without any further neighbours in any direction), use the function
        # information as the bounds for the heights.
        indices_without_neighbours = generate_indices_without_neighbours(self.no_points_per_axis)

        for index in indices_without_neighbours:
            l_b_constraints[index] = self.function_info[index][0]
            u_b_constraints[index] = self.function_info[index][1]

        return (l_b_constraints, u_b_constraints) 


    def build_function_coef_matrix_and_column_vector(self):
        ones = list(itertools.repeat(1.0, self.no_vars))
        ones_matrix = spmatrix(ones, range(self.no_vars), range(self.no_vars))
        minus_ones_matrix = -spmatrix(ones, range(self.no_vars), range(self.no_vars))
        
        coef_matrix = sparse([ones_matrix, minus_ones_matrix])

        (l_b_constraints, u_b_constraints) = self.build_constraints_from_function_info()
        flat_l_b_constraints = matrix(l_b_constraints.flatten())
        flat_u_b_constraints = matrix(u_b_constraints.flatten())
        
        # l <= x constraint will be rewritten as -x <= -l.
        column_vector = matrix([flat_u_b_constraints, -flat_l_b_constraints])

        return (coef_matrix, column_vector)


    def build_constraints_from_derivative_info(self):
        coords_ignoring_last_point \
            = [generate_indices_ignoring_last_coordinate_on_axis(self.no_points_per_axis, i) \
               for i in range(self.n)]

        # Number of elements for each of the upper/lower bound column vectors.
        no_elems_b_vec = sum([len(x) for x in coords_ignoring_last_point]) 
        
        l_b_constraints = np.empty(no_elems_b_vec)
        u_b_constraints = np.empty(no_elems_b_vec)

        index = 0
       
        for ith_partial, coords in enumerate(coords_ignoring_last_point):
            for coord in coords:
                next_index = coord[ith_partial] + 1
                current_index = coord[ith_partial]
                grid_diff = self.grid_info[ith_partial][next_index] - \
                            self.grid_info[ith_partial][current_index]
                l_b_constraints[index] = grid_diff * self.derivative_info[coord][ith_partial][0]
                u_b_constraints[index] = grid_diff * self.derivative_info[coord][ith_partial][1]
                index = index + 1

        return (l_b_constraints, u_b_constraints)


    def build_derivative_coef_matrix_and_column_vector(self):
        block_heights = calculate_block_heights(self.no_points_per_axis)
        adjacent_offsets = calculate_adjacent_sub_block_offsets(self.no_vars, 
                                                                self.no_points_per_axis)
        matrices_list = []

        for index, block_height in enumerate(block_heights):
            distance = calculate_distance_between_non_zero_entries(index, self.no_points_per_axis)
            no_sub_blocks = calculate_number_of_sub_blocks(index, self.no_points_per_axis)
            matrices_list.append(build_matrix_for_partial_derivative(adjacent_offsets[index],
                                 block_height, no_sub_blocks, distance))

        upper_half_matrix = sparse(matrices_list)
        coef_matrix = sparse([upper_half_matrix, -upper_half_matrix])
        
        (l_b_constraints, u_b_constraints) = self.build_constraints_from_derivative_info()
        flat_l_b_constraints = matrix(l_b_constraints)
        flat_u_b_constraints = matrix(u_b_constraints)

        # l <= x constraint will be rewritten as -x <= -l.
        column_vector = matrix([flat_u_b_constraints, -flat_l_b_constraints])

        return (coef_matrix, column_vector)


    def plot_3D_objects_for_2D_case(self):
        if self.n != 2:
            print "Plotting is only available for 2D domain."
            return

        grid_points = generate_indices(self.no_points_per_axis, False)
        x, y = [list(tup) for tup in zip(*grid_points)]
        x = [self.grid_info[0][index] for index in x]
        y = [self.grid_info[1][index] for index in y]
        min_z = self.min_heights.flatten()
        max_z = self.max_heights.flatten()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # Do not use Delaunay triangulation. Instead, generate the labels of the triangles in 
        # counter-clockwise fashion and supply there labels to the plot_trisurf function call.
        triangles = []

        x_dim = self.no_points_per_axis[0]
        y_dim = self.no_points_per_axis[1]

        for i in range(x_dim - 1):
            for j in range(y_dim - 1):
                label = y_dim * i + j
                triangles.append([label, label + y_dim, label + 1])
                triangles.append([label + y_dim, label + y_dim + 1, label + 1])

        ax.plot_trisurf(x, y, min_z, cmap='Blues', linewidth=0.5, antialiased=False, 
                        triangles=triangles)
        ax.plot_trisurf(x, y, max_z, cmap='Reds', linewidth=0.5, antialiased=False, 
                        triangles=triangles)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')

        plt.xticks(self.grid_info[0])
        plt.yticks(self.grid_info[1])

        ax.set_zlim(0.95 * min(min_z) , max(max_z) * 1.05)

        plt.show()


    def plot_polynomial(self):
        grid_points = generate_indices(self.no_points_per_axis, False)
        x, y = np.meshgrid(self.grid_info[0], self.grid_info[1])

        polynomial = build_polynomial([(1, 3, 1), (-1, 0, 3), (2, 2, 0)])
        
        z = []
        for grid_point in grid_points:
            x_coord = self.grid_info[0][grid_point[0]]
            y_coord = self.grid_info[1][grid_point[1]]
            point = (x_coord, y_coord)
            z.append(float(polynomial.eval(point)))

        z = np.array(z).reshape((self.no_points_per_axis[0], self.no_points_per_axis[1]))

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(x, y, z, cmap='Blues', linewidth=0.5, antialiased=False)

        plt.show()        


    def display_LP_problem(self, coef_matrix, vector):
        grid_indices = generate_indices(self.no_points_per_axis, False)
        no_rows = coef_matrix.size[0]
        no_cols = coef_matrix.size[1]
        no_d_constraints = (no_rows - 2 * self.no_vars) / 2

        print "LP problem:"

        for row in range(no_rows):
            non_zeros_indices = [(row + col * no_rows) for col in range(no_cols) \
                                  if coef_matrix[row + col * no_rows] != 0.0]
            # List of type [(value, true_row, true_col)], where value is either 1 or -1.
            true_non_zero_indices = [(coef_matrix[non_zeros_index], row, non_zeros_index / no_rows)\
                                     for non_zeros_index in non_zeros_indices]
            terms = []

            for true_non_zero_index in true_non_zero_indices:
                is_one = true_non_zero_index[0] == 1.0
                tup = grid_indices[true_non_zero_index[2]]
                term = 'h_' + ','.join([str(t) for t in tup])
                if not is_one:
                    term = '- ' + term

                terms.append(term)

            if len(true_non_zero_indices) == 1:
                if row < self.no_vars:
                    print str(-vector[row + self.no_vars]) + \
                        ' <= ' + terms[0] + ' <= ' + str(vector[row])
            else:
                if row < no_rows - no_d_constraints: 
                    print str(-vector[row + no_d_constraints]) + \
                        ' <= ' + terms[1] + ' ' + terms[0] + ' <= ' + str(vector[row])

####################################################################################################
##################################### COMMAND LINE ARGUMENTS #######################################
####################################################################################################

def command_line_arguments():
    usage = """

    consistency.py:

    Purpose: this script will check if the pair of step functions (f, g), where f : U -> IR and 
    g : U -> IR^n (the function and derivative information, respectively) are consistent, i.e. if 
    there exists a third map 'h' that is approximated by the first component of the pair ('f') and 
    whose derivative is approximated by the second component of the pair ('g'). Furthermore, if
    such a map exists, the script will return the least and greatest such maps, that have the added
    property of being piece-wise linear.

    Usage:

        python consistency.py --input_file <path_to_file>
    
    """

    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input-file", dest="input_file",
                      help="Specifies the path to the input file.")
    parser.add_option("-d", "--dimension", dest="dimension", type='int',    
                      help="Specifies the dimension of the function domain.")
    parser.add_option("-p", "--no-points-per-axis", dest="no_points_per_axis", type='int',
                      help="Specifies the number of points along each axis that is used for"
                           "automatically generating input files.")
    parser.add_option("-g", "--generate-input", dest="generate_input", action="store_true",
                      help="Specifies whether automatic input is generated to test consistency.")
    parser.add_option("-P", "--from-poly", dest="from_poly", action="store_true",
                      help="Specifies whether automatic input is generated from a polynomial.")

    return parser.parse_args()


def main():
    (options, args) = command_line_arguments()

    if options.generate_input:
        generate_test_file(options.input_file, options.dimension, options.no_points_per_axis, 
                           options.from_poly)
    
    cons = Consistency(options.input_file)
    cons.solve_LP_problem()
    cons.plot_3D_objects_for_2D_case()

if __name__ == '__main__':
    main()
