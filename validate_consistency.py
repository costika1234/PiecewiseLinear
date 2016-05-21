#/usr/local/bin/python

from consistency import Consistency
from input_generator import InputGenerator

import numpy as np

PATH_TO_FILE = 'test.txt'

def main():
    dimensions = [2, 3, 4, 5]
    epsilons = [0, 0.0001, 100.00]

    for dim in dimensions:
        random_numbers = np.random.randint(2, 5, dim).tolist()
        no_points_per_axis = ' '.join([str(no) for no in random_numbers])

        for eps in epsilons:
            for from_poly in [True, False]:
                input_gen = InputGenerator(PATH_TO_FILE, dim, no_points_per_axis, from_poly, eps)
                input_gen.generate_test_file()
                random_heights = input_gen.random_heights
                
                cons = Consistency(PATH_TO_FILE, random_heights, plot_surfaces=False)
                if cons.solve_LP_problem() is False:
                    print "Counterexample found!"
                    return 


if __name__ == '__main__':
    main()
