#/usr/local/bin/python

from consistency import Consistency
from input_generator import InputGenerator
from optparse import OptionParser
from utils import Utils
from timeit import default_timer as timer

import numpy as np
import sys

PATH_TO_FILE = 'test.txt'

class Measurements:

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def run(self):
        for dim in self.dimensions:
            TOTAL_TIME  = timer()
            INPUT_TIME  = timer()

            no_points_per_axis = [2] * dim
            no_points_per_axis = ' '.join([str(no) for no in no_points_per_axis])

            input_gen = InputGenerator(PATH_TO_FILE,
                                       True,
                                       dim,
                                       no_points_per_axis,
                                       rand_points_axis=True,
                                       from_poly=False,
                                       eps=0.0)

            input_gen.generate_test_file()

            INPUT_TIME = timer() - INPUT_TIME

            cons = Consistency(PATH_TO_FILE, input_gen, False, False, False)
            (MATRIX_TIME, LP_1_TIME, LP_2_TIME) = cons.solve_LP_problem()

            TOTAL_TIME = timer() - TOTAL_TIME

            print (INPUT_TIME, MATRIX_TIME, LP_1_TIME, LP_2_TIME, \
                   INPUT_TIME + MATRIX_TIME + LP_1_TIME + LP_2_TIME, TOTAL_TIME)


def main():
    print "Measurements:"
    dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    m = Measurements(dimensions)
    m.run()


if __name__ == '__main__':
    main()