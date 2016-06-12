#/usr/local/bin/python

from consistency import Consistency
from input_generator import InputGenerator
from optparse import OptionParser
from utils import Utils

import numpy as np
import sys

PATH_TO_FILE = 'test.txt'

class ConsistencyValidator:

    def __init__(self, dimensions, epsilons, is_input_consistent):
        self.dimensions = dimensions
        self.epsilons = epsilons
        self.is_input_consistent = is_input_consistent


    def run_tests(self):
        is_input_consistent_str = 'consistent'
        if not self.is_input_consistent:
            is_input_consistent_str = 'in' + is_input_consistent_str

        print 'Start testing for %s input...\n' % is_input_consistent_str

        for dimension in self.dimensions:
            random_no_points_per_axis = np.random.randint(2, 12, dimension).tolist()
            no_points_per_axis = ' '.join([str(no) for no in random_no_points_per_axis])
            no_points_per_axis_display = ', '.join([str(no) for no in random_no_points_per_axis])

            print '    Testing for dimension %d with [%s] points on axis...' % \
                (dimension, no_points_per_axis_display)

            for epsilon in self.epsilons:
                print '        Testing for epsilon: ', epsilon
                for from_poly in [True, False]:
                    input_gen = InputGenerator(PATH_TO_FILE,
                                               self.is_input_consistent,
                                               dimension,
                                               no_points_per_axis,
                                               rand_points_axis=True,
                                               from_poly=from_poly,
                                               eps=epsilon)
                    input_gen.generate_test_file()

                    # Silence the print statements to stdout.
                    with Utils.nostdout():
                        cons = Consistency(PATH_TO_FILE, input_gen, False, False, True, None)
                        result = cons.solve_LP_problem()

                    if result is not self.is_input_consistent:
                        raise RuntimeError('Counterexample found, aborting. See the file %s '
                                           'for details.' % PATH_TO_FILE)

        print '\n...Finished testing. No counterexamples found.\n'


def command_line_arguments():
    usage = '''

    validate_consistency.py:

    Purpose: this script will test the implementation of the linear programming algorithm of
    consistency, for either consistent or inconsistent input, respectively. This will ensure that
    the algorithm will never classify inconsitent input as consistent or the other way around.

    Usage:

        * Testing for consistent input:
            python validate_consistency.py --cons-input

        * Testing for inconsistent input:
            python validate_consistency.py --incons-input

    '''

    parser = OptionParser(usage=usage)
    parser.add_option('-c', '--cons-input', dest='cons_input', action='store_true',
                      help='Specifies whether consistent input will be tested.')
    parser.add_option('-i', '--incons-input', dest='incons_input', action='store_true',
                      help='Specifies whether inconsistent input will be tested.')

    return parser.parse_args()


def validate_options(options):
    if options.cons_input and options.incons_input or \
       not options.cons_input and not options.incons_input:
        raise RuntimeError('Should specify only one of the following available options: '
                           '--cons-input for consistent input or --incons-input for inconsistent'
                           'input.')


def main():
    (options, args) = command_line_arguments()
    validate_options(options)

    dimensions = [2, 3, 4]
    epsilons = [0.0, 0.1, 0.5, 1.0, 10.0, 50.0]
    is_input_consistent = options.cons_input is not None

    cons_validator = ConsistencyValidator(dimensions, epsilons, is_input_consistent)
    cons_validator.run_tests()


if __name__ == '__main__':
    main()

