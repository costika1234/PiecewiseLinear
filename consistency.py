#/usr/local/bin/python
 
from cvxopt import matrix, solvers, sparse, spmatrix
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
from input_generator import InputGenerator
from parser import Parser
from utils import Utils

import itertools
import matplotlib.pyplot as plt
import numpy as np
import re
import sympy as sp
import sys

class Consistency:

    def __init__(self, input_file, random_heights):
        # Domain dimension (integer).
        self.n = Parser.init_dimension(input_file)

        # Grid information (numpy array).
        self.grid_info = Parser.init_grid_info(input_file, self.n)

        # Number of points on each of the 'n' axis (list).
        self.no_points_per_axis = Parser.init_no_points_per_axis(self.grid_info)

        # Function information (n-dim numpy array of pairs).
        self.function_info = Parser.init_function_info(input_file, 
                                                       self.n, 
                                                       self.no_points_per_axis)
        
        # Derivative information (n-dim numpy array of tuples).
        self.derivative_info = Parser.init_derivative_info(input_file, 
                                                           self.n, 
                                                           self.no_points_per_axis)

        # Minimum heights for least witness of consistency (n-dim numpy array).
        self.min_heights = np.zeros(self.no_points_per_axis)

        # Maximum heights for greatest witness of consistency (n-dim numpy array).
        self.max_heights = np.zeros(self.no_points_per_axis)

        # Number of decision variables (integer).
        self.no_vars = np.prod(self.no_points_per_axis)

        # Randomly generated heights to enable automatic testing (n-dim numpy array).
        self.random_heights = random_heights


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
            print 'No witness for consistency found.'


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


    def get_partial_derivatives_end_points(self, dimension):
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


    def build_constraints_from_derivative_info(self):
        indices_allowing_neighbours = Utils.get_grid_indices(self.no_points_per_axis, True)

        partial_derivatives_end_points = self.get_partial_derivatives_end_points(self.n)

        print partial_derivatives_end_points

        # for grid_index in indices_allowing_neighbours:
        #     print grid_index

        #     next_grid_indices = Utils.get_grid_indices_neighbours(grid_index)
        #     print next_grid_indices

        #     for ith_partial in range(self.n):
        #         for end_points in partial_derivatives_end_points[ith_partial]:
        #             next_index = next_grid_indices[end_points[1]]
        #             curr_index = next_grid_indices[end_points[0]]

        #             print next_index, curr_index
        #         print "--------"
        

        ### NEW STUFF

        no_rows_for_partial_derivatives = Utils.calculate_block_heights(self.no_points_per_axis)

        # print no_rows_for_partial_derivatives

        u_b_constraints = []
        l_b_constraints = []

        for ith_partial in range(self.n):
            partial_l_b_constraints = np.empty(Utils.get_dimension_for_row_vector(
                self.no_points_per_axis, ith_partial))
            partial_l_b_constraints.fill(float('-inf'))

            partial_u_b_constraints = np.empty(Utils.get_dimension_for_row_vector(
                self.no_points_per_axis, ith_partial))
            partial_u_b_constraints.fill(float('inf'))

            for grid_index in indices_allowing_neighbours:
                print grid_index

                next_grid_indices = Utils.get_grid_indices_neighbours(grid_index)

                for end_points in partial_derivatives_end_points[ith_partial]:
                    next_index = next_grid_indices[end_points[1]]
                    curr_index = next_grid_indices[end_points[0]]

                    print " --> ", next_index, curr_index

                    next_axis_index = next_index[ith_partial]
                    curr_axis_index = curr_index[ith_partial]

                    grid_diff = self.grid_info[ith_partial][next_axis_index] - \
                                self.grid_info[ith_partial][curr_axis_index]

                    partial_l_b_constraints[curr_index] = max(partial_l_b_constraints[curr_index], \
                        grid_diff * self.derivative_info[grid_index][ith_partial][0])
                    partial_u_b_constraints[curr_index] = min(partial_u_b_constraints[curr_index], \
                        grid_diff * self.derivative_info[grid_index][ith_partial][1])

                    #print curr_index 

            l_b_constraints.extend(partial_l_b_constraints.flatten().tolist())
            u_b_constraints.extend(partial_u_b_constraints.flatten().tolist())


        # print np.array(u_b_constraints)
        #print "------------------------------------"

        ## END NEEW STUFF


        # grid_indices_excluding_last_point \
        #     = [Utils.get_grid_indices_excluding_last_point_on_axis(self.no_points_per_axis, i) \
        #        for i in range(self.n)]

        # # Number of elements for each of the upper/lower bound column vectors.
        # no_elems_b_vec = sum([len(x) for x in grid_indices_excluding_last_point]) 
        
        # # print "no_elems_b_vec =", no_elems_b_vec
        # # print grid_indices_excluding_last_point

        # l_b_constraints_2 = np.empty(no_elems_b_vec)
        # u_b_constraints_2 = np.empty(no_elems_b_vec)

        # index = 0
       
        # for ith_partial, coords in enumerate(grid_indices_excluding_last_point):
        #     print coords
        #     for coord in coords:
        #         print coord
        #         next_index, current_index = coord[ith_partial] + 1, coord[ith_partial]
        #         grid_diff = self.grid_info[ith_partial][next_index] - \
        #                     self.grid_info[ith_partial][current_index]
        #         l_b_constraints_2[index] = grid_diff * self.derivative_info[coord][ith_partial][0]
        #         u_b_constraints_2[index] = grid_diff * self.derivative_info[coord][ith_partial][1]
        #         index = index + 1

        #         #print coord

        # print np.array(u_b_constraints)
        # print "---------------------------"
        # print u_b_constraints_2
        # print "---------------------------"
        # print u_b_constraints_2 - np.array(u_b_constraints)

        # import sys
        # sys.exit()

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

        fig = plt.figure()
        fig.canvas.set_window_title('Consistency')
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

        if self.random_heights is not None:
            ax.plot_trisurf(x, y, self.random_heights.flatten(), cmap='Greens', linewidth=0.5, 
                            antialiased=False, triangles=triangles)

        ax.set_zlim(min(min_z) - 1.0, max(max_z) + 1.0)

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

####################################################################################################
##################################### COMMAND LINE ARGUMENTS #######################################
####################################################################################################

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

        * Automatic mode which generates consistent pairs (f, g) randomly and then runs the
          algorithm to check that consistency holds. One can specify the optional '--from-poly'
          flag to indicate that the function value is obtained from a piece-wise linear polynomial.

            python consistency.py --input_file <path_to_file>
                                  --generate-input
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
                      help='Specifies the number of points along each axis, as a string of'
                           'space-separated values. The number of points on each dimension will'
                           'divide each axis in equal segments and thus create the required grid.')
    parser.add_option('-g', '--generate-input', dest='generate_input', action='store_true',
                      help='Specifies whether automatic input is generated to test consistency.')
    parser.add_option('', '--from-poly', dest='from_poly', action='store_true',
                      help='Specifies whether automatic input is generated from a polynomial.')
    parser.add_option('-e', '--epsilon', dest='epsilon', type='float',
                      help='Specifies the tolerance value for generating random intervals for '
                            'function and derivative information, respectively. When set to 0, '
                            'both the least, greatest and original function information will '
                            'coincide.')

    return parser.parse_args()


def validate_options(options):
    if options.generate_input:
        if options.dimension is None or \
           options.no_points_per_axis is None or \
           options.epsilon is None:
            raise RuntimeError('Dimension, number of points per axis and epsilon must be specified '
                               'when generating automatic input.')

        if options.dimension < 2:
            raise RuntimeError('Invalid domain dimension. Should be greater than or equal to 2.')

        if len(options.no_points_per_axis.split(' ')) != options.dimension:
            raise RuntimeError('Invalid number of points per axis.')


def main():
    (options, args) = command_line_arguments()
    validate_options(options)
    random_heights = None

    if options.generate_input: 
        input_gen = InputGenerator(options.input_file,
                                   options.dimension,
                                   options.no_points_per_axis,
                                   options.from_poly,
                                   options.epsilon)
        input_gen.generate_test_file()
        random_heights = input_gen.random_heights

    cons = Consistency(options.input_file, random_heights)
    cons.solve_LP_problem()
    cons.plot_3D_objects_for_2D_case()


if __name__ == '__main__':
    main()

