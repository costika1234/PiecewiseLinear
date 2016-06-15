#/usr/local/bin/python

from consistency import Consistency
from input_generator import InputGenerator
from optparse import OptionParser
from utils import Utils
from timeit import default_timer as timer

import numpy as np
import sys
import matplotlib.pyplot as plt

PATH_TO_FILE = 'test.txt'

class Measurements:

    def __init__(self, dimensions):
        self.dimensions = dimensions

    def run(self):
        times = []

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

            #TOTAL_TIME = INPUT_TIME + MATRIX_TIME + LP_1_TIME + LP_2_TIME

            times.append((INPUT_TIME, MATRIX_TIME, LP_1_TIME, LP_2_TIME))


        return times

def sumzip(*items):
    return [sum(values) for values in zip(*items)]

def main():
    MIN_DIM = 2
    MAX_DIM = 10
    NO_DIMS = MAX_DIM - MIN_DIM
    dimensions = range(MIN_DIM, MAX_DIM, 1)
    m = Measurements(dimensions)

    results = []
    N = 10

    # Run several experiments.
    for i in range(N):
        results.append(list(m.run()))


    input_time = []
    matrix_time = []
    lp1_time = []
    lp2_time = []
    #total_time = []

    for dim in dimensions:
        input_sum = 0
        matrix_sum = 0
        lp1_sum = 0
        lp2_sum = 0
        #total_sum = 0
        for i in range(N):
            input_sum += results[i][dim-2][0]
            matrix_sum += results[i][dim-2][1]
            lp1_sum += results[i][dim-2][2]
            lp2_sum += results[i][dim-2][3]
            #total_sum += results[i][dim-2][4]

        input_time.append(input_sum / float(N))
        matrix_time.append(matrix_sum / float(N))
        lp1_time.append(lp1_sum / float(N))
        lp2_time.append(lp2_sum / float(N))


    i = tuple(input_time)
    m = tuple(matrix_time)
    l1 = tuple(lp1_time)
    l2 = tuple(lp2_time)



    ind = np.arange(NO_DIMS)    # the x locations for the groups
    width = 0.5      # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, i, width, color='#f39c12')
    p2 = plt.bar(ind, m, width, color='#16a085', bottom=sumzip(i))
    p3 = plt.bar(ind, l1, width, color='#2980b9', bottom=sumzip(i, m))
    p4 = plt.bar(ind, l2, width, color='#e74c3c', bottom=sumzip(i, m, l1))


    plt.ylabel('Time (s)')
    plt.xlabel('Dimension')
    plt.title('Performance Analysis')

    plt.xticks(ind + width/2., tuple([str(no) for no in dimensions]))
    plt.xlim(-0.5, 8)
    #plt.yticks(np.arange(0, 81, 10))

    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Input', 'Matrix form', 'Min LP', 'Max LP'), loc='upper left')

    #plt.show()

    plt.savefig("performan4ce.eps", format="eps", dpi=1000)



if __name__ == '__main__':
    main()