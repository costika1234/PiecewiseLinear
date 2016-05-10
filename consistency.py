#/usr/local/bin/python
 
from cvxopt import matrix, solvers, sparse, spmatrix, printing
from operator import mul
from optparse import OptionParser

import itertools
import re
import numpy as np
import sys


LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'

GRID_INFO_STRING       = r'# Grid information'
FUNCTION_INFO_STRING   = r'# Function information'
DERIVATIVE_INFO_STRING = r'# Derivative information'

FLOAT_NUMBER_REGEX = r'[-+]?[0-9]*\.?[0-9]+'
DIMENSION_REGEX    = r'# Dimension: (\d+)'

####################################################################################################
######################################### DATA GENERATION ##########################################
####################################################################################################

def get_zipped_list(no_elements):
    # Returns a list of the form: [(a, b), ...] to help generating dummy data.
    l = np.array([round(x, 2) for x in np.linspace(no_elements, 1.0, no_elements)])
    u = l + no_elements
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


def generate_tuples_info(file_descriptor, n, no_points_per_axis, is_function_info):
    no_elements = np.prod(no_points_per_axis)
    dt = get_dtype(n, is_function_info)

    if is_function_info:
        # Create flat array with pairs (c-, c+).
        flat_function_info = get_zipped_list(no_elements)
        nd_array = np.array(flat_function_info, dtype=dt).reshape(no_points_per_axis)
    else:
        # Create flat array with tuples ((c1-, c1+), (c2-, c2+), ...).
        zipped = get_zipped_list(no_elements)
        flat_derivative_info = zip(*[zipped for _ in range(n)])
        nd_array = np.array(flat_derivative_info, dtype=dt).reshape(no_points_per_axis)

    # Write contents to file.`
    file_descriptor.write('# Array shape: {0}\n'.format(nd_array.shape))
    traverse_nd_array(nd_array, file_descriptor, n)


def generate_test_file(input_file, n, no_points_per_axis):
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
        
        # Function information.
        f.write('\n# Function information (specified as a %d-dimensional array of intervals, where '
                'an entry is of the form (c-, c+), c- < c+, and represents the constraint for the '
                'function value at a particular grid point):\n' % n)
        generate_tuples_info(f, n, no_points_per_axis, is_function_info=True)

        # Derivative information.
        f.write('\n# Derivative information (specified as a %d-dimensional array of tuples of '
                'intervals, where an entry is of the form ((c1-, c1+), (c2-, c2+), ...), ci- < ci+,'
                ' and represents the constraints along each partial derivative at a '
                'particular grid point):\n' % n)
        generate_tuples_info(f, n, no_points_per_axis, is_function_info=False)

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

    #print minus_ones_and_ones
    #print row_range
    #print column_range
    print spmatrix(minus_ones_and_ones, row_range, column_range)

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

        # Heights information (n-dim numpy array).
        self.h = np.zeros(self.no_points_per_axis)

        # Number of decision variables (integer).
        self.no_vars = np.prod(self.no_points_per_axis)


    def build_LP_problem(self):
        #print self.n
        #print self.grid_info
        #print self.no_points_per_axis
        #print self.function_info
        #print self.derivative_info
        ones = list(itertools.repeat(1.0, self.no_vars))
        minus_ones = list(itertools.repeat(-1.0, self.no_vars))
        
        ones_matrix = spmatrix(ones, range(self.no_vars), range(self.no_vars))
        minus_ones_matrix = spmatrix(minus_ones, range(self.no_vars), range(self.no_vars))

        coefficient_matrix = sparse([ones_matrix, minus_ones_matrix])
        objective_function_vector = matrix(ones)

        (lower, upper) = self.build_constraints_from_function_info()
        (d_matrix, d_lower, d_upper) = self.build_constraints_from_derivative_info()

        print "LOWER BOUNDS:"
        print lower

        print "UPPER BOUNDS:"
        print upper

        upper_vector = matrix(upper.flatten())
        lower_vector = matrix(lower.flatten())
        # Need to add a minus to the lower bounds, as l <= x will be converted to -x <= - b.
        bounds_vector = matrix([upper_vector, -lower_vector])

        d_upper_vector = matrix(d_upper)
        d_lower_vector = matrix(d_lower)
        d_bounds_vector = matrix([d_upper_vector, -d_lower_vector])

        d_coef_matrix = matrix([d_matrix, d_matrix])

        #print coefficient_matrix
        #print bounds_vector
        #print "============================"
        #print d_coef_matrix
        #print d_bounds_vector

        final_coef_matrix = matrix([coefficient_matrix, d_coef_matrix])
        final_vector = matrix([bounds_vector, d_bounds_vector])

        #print len(coefficient_matrix)
        #print len(d_coef_matrix)
        #print len(final_coef_matrix)
        #print len(final_vector)

        #print coefficient_matrix
        #print bounds_vector
        #print objective_function_vector

        # Solve the LP problem (combine constraints for both function and derivative info).
        sol = solvers.lp(objective_function_vector, final_coef_matrix, final_vector)
        
        print sol['x']

        #for index in range(self.no_vars):
        #    print sol['x'][index]


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


    def build_constraints_from_derivative_info(self):
        block_heights = calculate_block_heights(self.no_points_per_axis)
        adjacent_offsets = calculate_adjacent_sub_block_offsets(self.no_vars, self.no_points_per_axis)
        matrices_list = []

        for index, block_height in enumerate(block_heights):
            print "index =", index
            distance = calculate_distance_between_non_zero_entries(index, self.no_points_per_axis)
            no_sub_blocks = calculate_number_of_sub_blocks(index, self.no_points_per_axis)
            # print "no_sub_blocks =", no_sub_blocks
            matrices_list.append(build_matrix_for_partial_derivative(adjacent_offsets[index], block_height, no_sub_blocks, distance))

        coef_matrix = matrix(matrices_list)

        coords_ignoring_last_point = [generate_indices_ignoring_last_coordinate_on_axis(self.no_points_per_axis, i) \
                                      for i in range(self.n)]

        # Number of elements for each of the upper/lower bound column vectors.
        no_elems_b_vec = sum([len(x) for x in coords_ignoring_last_point]) 
        
        l_b_constraints = np.empty(no_elems_b_vec)
        u_b_constraints = np.empty(no_elems_b_vec)

        index = 0
       
        for ith_partial, coords in enumerate(coords_ignoring_last_point):
            for coord in coords:
                l_b_constraints[index] = self.derivative_info[coord][ith_partial][0]
                u_b_constraints[index] = self.derivative_info[coord][ith_partial][1]
                index = index + 1

        return (coef_matrix, np.array(l_b_constraints), np.array(u_b_constraints))

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
    parser.add_option("", "--input-file", dest="input_file",
                      help="Specifies the path to the input file.")
    parser.add_option("", "--dimension", dest="dimension",
                      help="Specifies the dimension of the function domain.")
    
    return parser.parse_args()


def main():
    (options, args) = command_line_arguments()

    n = int(options.dimension)
    no_points_per_axis = tuple([x + 3 for x in range(n)])
    printing.options['width'] = 30
    
    #generate_test_file(options.input_file, n, no_points_per_axis)
    cons = Consistency(options.input_file)
    cons.build_LP_problem()


if __name__ == '__main__':
    main()
