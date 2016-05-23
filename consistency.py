#/usr/local/bin/python

from cvxopt import matrix, solvers, sparse, spmatrix
from input_generator import InputGenerator
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from parser import Parser
from utils import Utils

import itertools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import re
import sympy as sp
import sys

class Consistency:

    def __init__(self, input_file, plot_surfaces, input_generator):
        if input_generator is not None:
            self.n = input_generator.n
            self.grid_info = input_generator.grid_info
            self.no_points_per_axis = input_generator.no_points_per_axis
            self.function_info = input_generator.function_info
            self.derivative_info = input_generator.derivative_info
            self.random_heights = input_generator.random_heights
        else:
            self.n = Parser.init_dimension(input_file)
            self.grid_info = Parser.init_grid_info(input_file, self.n)
            self.no_points_per_axis = Parser.init_no_points_per_axis(self.grid_info)
            self.function_info = Parser.init_function_info(input_file, self.n,
                                                           self.no_points_per_axis)
            self.derivative_info = Parser.init_derivative_info(input_file, self.n,
                                                               self.no_points_per_axis)
            self.random_heights = None

        # Number of decision variables (integer).
        self.no_vars = np.prod(self.no_points_per_axis)

        # Minimum heights for least witness of consistency (n-dim numpy array).
        self.min_heights = np.zeros(self.no_points_per_axis)

        # Maximum heights for greatest witness of consistency (n-dim numpy array).
        self.max_heights = np.zeros(self.no_points_per_axis)

        # Flag which specifies whether the surfaces will be plotted for the 2D case.
        self.plot_surfaces = plot_surfaces


    def solve_LP_problem(self):
        (f_coef_matrix, f_column_vector) = self.build_function_coef_matrix_and_column_vector()
        (d_coef_matrix, d_column_vector) = self.build_derivative_coef_matrix_and_column_vector()

        # Solve the LP problem by combining constraints for both function and derivative info.
        objective_function_vector = matrix(list(itertools.repeat(1.0, self.no_vars)))
        coef_matrix = sparse([f_coef_matrix, d_coef_matrix])
        column_vector = matrix([f_column_vector, d_column_vector])

        min_sol = solvers.lp(objective_function_vector, coef_matrix, column_vector)
        is_consistent = min_sol['x'] is not None

        # Print the LP problem for debugging purposes.
        self.display_LP_problem(coef_matrix, column_vector)

        if is_consistent:
            self.min_heights = np.array(min_sol['x']).reshape(self.no_points_per_axis)
            print np.around(self.min_heights, decimals=2)

            # Since consistency has been established, solve the converse LP problem to get the
            # maximal bounding surface.
            max_sol = solvers.lp(-objective_function_vector, coef_matrix, column_vector)
            self.max_heights = np.array(max_sol['x']).reshape(self.no_points_per_axis)
            print np.around(self.max_heights, decimals=2)

            if self.plot_surfaces:
                self.plot_3D_objects_for_2D_case()

        else:
            print 'No witness for consistency found.'

        return is_consistent


    def build_constraints_from_function_info(self):
        # Construct bounds of the type: c_ij- <= h_st <= c_ij+, s = i,i+1, t = j,j+1.
        l_b_constraints = np.empty(self.no_points_per_axis)
        u_b_constraints = np.empty(self.no_points_per_axis)

        l_b_constraints.fill(np.float('-inf'))
        u_b_constraints.fill(np.float('inf'))

        # Derive the constraints on the heights based on the adjacent subrectangles.
        indices_allowing_neighbours = Utils.get_grid_indices(self.no_points_per_axis, True)

        for grid_index in indices_allowing_neighbours:
            next_grid_indices = Utils.get_grid_indices_neighbours(grid_index)
            l_b_function_value = self.function_info[grid_index][0]
            u_b_function_value = self.function_info[grid_index][1]

            for index in next_grid_indices:
                l_b_constraints[index] = max(l_b_constraints[index], l_b_function_value)
                u_b_constraints[index] = min(u_b_constraints[index], u_b_function_value)

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
        # Construct bounds of the type: b_ij_1- <= (h_i+1,j - h_ij) / (p_i+1 - p_i) <= b_ij_1+.
        indices_allowing_neighbours = Utils.get_grid_indices(self.no_points_per_axis, True)
        partial_derivatives_end_points = Utils.get_partial_derivatives_end_points(self.n)

        l_b_constraints = []
        u_b_constraints = []

        # Along each partial derivative and for all possible triangulations within a hyper-rectangle
        # ensure that we construct the tightest bounds for consecutive differences of heigths.
        for ith_partial in range(self.n):
            partial_l_b_constraints = np.empty(
                Utils.get_dimension_for_row_vector(self.no_points_per_axis, ith_partial))
            partial_u_b_constraints = np.empty(
                Utils.get_dimension_for_row_vector(self.no_points_per_axis, ith_partial))

            partial_l_b_constraints.fill(float('-inf'))
            partial_u_b_constraints.fill(float('inf'))

            for grid_index in indices_allowing_neighbours:
                next_grid_indices = Utils.get_grid_indices_neighbours(grid_index)

                for end_points in partial_derivatives_end_points[ith_partial]:
                    next_index = next_grid_indices[end_points[1]]
                    curr_index = next_grid_indices[end_points[0]]

                    next_coord = Utils.convert_grid_index_to_coord(next_index, self.grid_info)
                    curr_coord = Utils.convert_grid_index_to_coord(curr_index, self.grid_info)
                    grid_diff = next_coord[ith_partial] - curr_coord[ith_partial]

                    partial_l_b_constraints[curr_index] = max(partial_l_b_constraints[curr_index], \
                        grid_diff * self.derivative_info[grid_index][ith_partial][0])
                    partial_u_b_constraints[curr_index] = min(partial_u_b_constraints[curr_index], \
                        grid_diff * self.derivative_info[grid_index][ith_partial][1])

            l_b_constraints.extend(partial_l_b_constraints.flatten().tolist())
            u_b_constraints.extend(partial_u_b_constraints.flatten().tolist())

        return (l_b_constraints, u_b_constraints)


    def build_derivative_coef_matrix_and_column_vector(self):
        block_heights = Utils.calculate_block_heights(self.no_points_per_axis)
        adjacent_offsets = Utils.calculate_adjacent_sub_block_offsets(self.no_vars,
                                                                      self.no_points_per_axis)
        matrices_list = []

        for index, block_height in enumerate(block_heights):
            distance = Utils.calculate_distance_between_non_zero_entries(
                index, self.no_points_per_axis)
            no_sub_blocks = Utils.calculate_number_of_sub_blocks(index, self.no_points_per_axis)
            matrices_list.append(Utils.build_matrix_for_partial_derivative(
                adjacent_offsets[index], block_height, no_sub_blocks, distance))

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
            print 'Plotting is only available for 2D domain.'
            return

        grid_points = Utils.get_grid_indices(self.no_points_per_axis, False)
        x, y = [list(tup) for tup in zip(*grid_points)]
        x = [self.grid_info[0][index] for index in x]
        y = [self.grid_info[1][index] for index in y]
        min_z = self.min_heights.flatten()
        max_z = self.max_heights.flatten()

        fig = plt.figure(figsize=(14.4, 9), facecolor='#34495e')
        fig.canvas.set_window_title('Consistency')
        ax = fig.gca(projection='3d')
        ax.set_axis_bgcolor('#34495e')

        # Do not use Delaunay triangulation. Instead, generate the labels of the triangles in
        # counter-clockwise fashion and supply there labels to the plot_trisurf function call.
        x_dim = self.no_points_per_axis[0]
        y_dim = self.no_points_per_axis[1]
        triangles = Utils.get_triangulation(x_dim, y_dim)

        ax.plot_trisurf(x, y, min_z, cmap='Blues', linewidth=0.5, antialiased=False,
                        triangles=triangles)
        ax.plot_trisurf(x, y, max_z, cmap='Reds', linewidth=0.5, antialiased=False,
                        triangles=triangles)

        least_surface = mpatches.Patch(color='#3498db', label='Least Surface')
        greatest_surface = mpatches.Patch(color='#e74c3c', label='Greatest Surface')
        legend_handles = [greatest_surface, least_surface]

        if self.random_heights is not None:
            ax.plot_trisurf(x, y, self.random_heights.flatten(), cmap='Greens', linewidth=0.5,
                            antialiased=False, triangles=triangles)
            original_surface = mpatches.Patch(color='#27ae60', label='Original Surface')
            legend_handles = [greatest_surface, original_surface, least_surface]

        plt.legend(handles=legend_handles)

        ax.set_xlabel('\n\nX axis (%d points)' % x_dim, color='white')
        ax.set_ylabel('\n\nY axis (%d points)' % y_dim, color='white')
        ax.set_zlabel('\n\nZ axis', color='white')
        ax.set_zlim(min(min_z) - 1.0, max(max_z) + 1.0)

        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        plt.xticks(self.grid_info[0])
        plt.yticks(self.grid_info[1])
        plt.show()


    def display_LP_problem(self, coef_matrix, vector):
        grid_indices = Utils.get_grid_indices(self.no_points_per_axis, False)
        no_rows = coef_matrix.size[0]
        no_cols = coef_matrix.size[1]
        no_d_constraints = (no_rows - 2 * self.no_vars) / 2

        print 'LP problem:'

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


def command_line_arguments():
    usage = '''

    consistency.py:

    Purpose: this script will check if the pair of step functions (f, g), where f : U -> IR and
    g : U -> IR^n (the function and derivative information, respectively) are consistent, i.e. if
    there exists a third map 'h' that is approximated by the first component of the pair ('f') and
    whose derivative is approximated by the second component of the pair ('g'). Furthermore, if
    such a map exists, the script will return the least and greatest such maps, that have the added
    property of being piece-wise linear.

    Usage:

        * Manual mode for checking consistency based on input file.

            python consistency.py --input_file <path_to_file>

        * Automatic mode which generates random consistent/inconsistent pairs (f, g) and then runs
          the algorithm to determine whether consistency holds or not. One can specify the optional
          '--from-poly' flag to indicate that the function value is obtained from a piece-wise
          linear polynomial. In additional, a tolerance value ('epsilon') can be specified in order
          to control the distance between the minimal and maximal surfaces, respectively.

            python consistency.py --input_file <path_to_file>
                                  [--gen-cons-input | --gen-incons-input]
                                  --dimension <domain_dimension>
                                  --no-points-per-axis <no_points_per_axis> (e.g. '2 3 4')
                                  --epsilon <epsilon>
                                  [--from-poly]

    '''

    parser = OptionParser(usage=usage)
    parser.add_option('-i', '--input-file', dest='input_file', type='string',
                      help='Specifies the path to the input file.')
    parser.add_option('-d', '--dimension', dest='dimension', type='int',
                      help='Specifies the dimension of the function domain.')
    parser.add_option('-p', '--no-points-per-axis', dest='no_points_per_axis', type='string',
                      help='Specifies the number of points along each axis, as a string of '
                           'space-separated values. The number of points on each dimension will '
                           'divide each axis in equal segments and thus create the required grid.')
    parser.add_option('', '--gen-cons-input', dest='gen_cons_input', action='store_true',
                      help='Specifies whether randomly consistent input is generated.')
    parser.add_option('', '--gen-incons-input', dest='gen_incons_input', action='store_true',
                      help='Specifies whether randomly inconsistent input is generated.')
    parser.add_option('', '--from-poly', dest='from_poly', action='store_true',
                      help='Specifies whether automatic input is generated from a polynomial.')
    parser.add_option('', '--plot', dest='plot_surfaces', action='store_true',
                      help='Specifies whether surfaces will be plotted for the 2D case.')
    parser.add_option('-e', '--epsilon', dest='epsilon', type='float',
                      help='Specifies the tolerance value for generating random intervals for '
                           'function and derivative information, respectively.')

    return parser.parse_args()


def validate_options(options):
    if options.gen_cons_input and options.gen_incons_input:
        raise RuntimeError('Mutually exclusive options specified. Should either set generate '
                           'consistent input or generate inconsistent input, or none.')

    if options.gen_cons_input or options.gen_incons_input:
        if options.dimension is None or \
           options.no_points_per_axis is None or \
           options.epsilon is None:
            raise RuntimeError('Dimension, number of points per axis and epsilon must be specified '
                               'when generating automatic input.')

        if options.dimension < 2:
            raise RuntimeError('Invalid domain dimension. Should be greater than or equal to 2.')

        no_points_per_axis_list = [int(no) for no in options.no_points_per_axis.split(' ')]
        if len(no_points_per_axis_list) != options.dimension or \
           not all(no >= 2 for no in no_points_per_axis_list):
            raise RuntimeError('Invalid number of points per axis. The list must have as many '
                               'elements as the number of dimensions and each axis must contain at '
                               'least 2 points.')


def main():
    (options, args) = command_line_arguments()
    validate_options(options)
    input_generator = None

    if options.gen_cons_input or options.gen_incons_input:
        input_generator = InputGenerator(options.input_file,
                                         options.gen_cons_input is not None,
                                         options.dimension,
                                         options.no_points_per_axis,
                                         options.from_poly,
                                         options.epsilon)
        input_generator.generate_test_file()

    cons = Consistency(options.input_file, options.plot_surfaces, input_generator)
    cons.solve_LP_problem()


if __name__ == '__main__':
    main()

