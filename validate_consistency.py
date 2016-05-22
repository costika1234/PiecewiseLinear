#/usr/local/bin/python

from consistency import Consistency
from input_generator import InputGenerator

import contextlib
import numpy as np
import sys

PATH_TO_FILE = 'test.txt'


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def main():
    dimensions = [2, 3]
    epsilons = [0, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print 'Testing consistency for epsilons in %s...' % str(epsilons)

    for dim in dimensions:
        random_no_points_per_axis = np.random.randint(5, 15, dim).tolist()
        no_points_per_axis = ' '.join([str(no) for no in random_no_points_per_axis])
        no_points_per_axis_display = ', '.join([str(no) for no in random_no_points_per_axis])
        print '    Testing consistency in dimension %d with [%s] points on axis...' % \
            (dim, no_points_per_axis_display)

        for eps in epsilons:
            for from_poly in [True, False]:
                input_gen = InputGenerator(PATH_TO_FILE, dim, no_points_per_axis, from_poly, eps)
                input_gen.generate_test_file()

                # Silence the print statements to stdout.
                with nostdout():
                    cons = Consistency(PATH_TO_FILE, False, input_gen)
                    result = cons.solve_LP_problem()

                if result is False:
                    print "Counterexample found! See the file %s for details." % PATH_TO_FILE
                    return

    print "...Finished testing. Consistency holds for the randomly-generated input."


if __name__ == '__main__':
    main()
