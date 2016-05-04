#/usr/local/bin/python
 
from optparse import OptionParser

import re
import numpy as np

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound' 
FLOAT_NUMBER_REGEX  = r"[-+]?[0-9]*\.?[0-9]+"
FUNCTION_INFO_REGEX = r"\((?P<" + LOWER_BOUND + r">" + FLOAT_NUMBER_REGEX + r"),\s*" + \
                        r"(?P<" + UPPER_BOUND + r">" + FLOAT_NUMBER_REGEX + r")\)" 

def rescale_axis():
	# Rescale axis so that the map [a, b]^n -> R will be converted to 
	# [0, 1]^n -> R for simplicity. Basically, 
	#
	#    a <= x <= b
	#
	#  will be converted to:
	#
	#    0 <= (x - a) / (b - a) <= 1
	pass

def parse_input_file(input_file):
	# Parse the input file. Assuming maps [a, b]^n -> R, we define the following
	# format (note that [<...>] are regarded as tags to aid readibility of the
	# sections:
	#   
	#   [domain]
	#   
	#
	#   domain end-points ('a' and 'b')
	#
	#   followed by:   
	#
	#   'n' rows, each partioning the i-th coordinate axis, i = 0, n-1
	#
	#   followed by:
	#   
	#   Function information: d1 x d2 x ... x dn - dimensional array, where di =
	#   the number of partition points along the i-th coordinate axis. Each
	#   entry of this array should be given as a pair (c-, c+), that is, the
	#   interval that the function is allowed to lie at the point in the 
	#   resulting hyper-grid.
	#   
	#   followed by:
	#  
	#   Derivative information: as before, a d1 x d2 x ... x dn array, where
	#   at each hyper-grid point we have a tuple of the form: ([b1-, b1+], ...,
	#   [bn-, bn+]), i.e. each partial derivative is guaranteed to lie within an
	#   interval, which means that the derivative information is given as
	#   compact hyper-rectangles with sides parallel to the coordinate planes.
	#
	#   Throughout parsing the file, ensure that the intervals are well defined, 
	#   and that the hyper-grid partition is given in stricly increasing 
	#   monotonic order and coves the end-points of the provided fuction domain.
	pass


def generate_function_information(input_file, n, no_divisions_per_axis):
	no_elements = np.prod(no_divisions_per_axis) 
	a = np.arange(no_elements).reshape(no_divisions_per_axis)
	b = np.arange(no_elements)[::-1].reshape(no_divisions_per_axis)
	
	nd_array = np.array(zip(a.ravel(),b.ravel()), dtype=('f8,f8')).reshape(a.shape)

	# Write the array to disk
	with file(input_file, 'w+') as f:
	    f.write('# Array shape: {0}\n'.format(nd_array.shape))
	    traverse_nd_array(nd_array, f, n)


def traverse_nd_array(nd_array, f, depth):
	# Function which recursively traverses an n-dimenionsal array and saves the
	# array to file.
	if depth == 2:
		np.savetxt(f, nd_array, fmt='%s')
	else:
		for sub_nd_array in nd_array:
			traverse_nd_array(sub_nd_array, f, depth - 1)

	f.write('# End of %d depth \n' % depth)


def parse_function_information(input_file, no_divisions_per_axis):
	flat_nd_list = []

	with open(input_file) as f:
		for line in f:
			# Ignore possible comments in the input lines.
			if line.startswith('#'):
				continue

			# Append the pairs of lower and upper bounds to the flat list.
			for m in re.finditer(FUNCTION_INFO_REGEX, line):
				flat_nd_list.append((m.group(LOWER_BOUND), m.group(UPPER_BOUND)))

	# Finally, convert to the shape of an n-dimensional array from the given divisions.
	dt = ('float64, float64')
	return np.array(flat_nd_list, dtype=dt).reshape(no_divisions_per_axis) 


def command_line_arguments():
    usage = """

    consistency.py:

   	Purpose: this script will check if the pair of step functions (f, g), where f : U -> IR and 
   	g : U -> IR^n (the function and derivative information, respectively) are consistent, i.e. if 
   	there exists a third map 'h' that is approximated by the first component of the pair ('f') and 
   	whose derivative is approximated by the second compenent of the pair ('g'). Furthermore, if
   	such a map exists, the script will return the least and greatest such maps, that have the added
   	property of being piece-wise linear.

   	Usage:

   	    python consistency.py --input_file <path_to_file>
    
    """

    parser = OptionParser(usage=usage)

    parser.add_option("", "--input-file", dest="input_file",
                      help="Specifies the path to the input file.")

    return parser.parse_args()


def main():
	(options, args) = command_line_arguments()

	n = 2
	no_divisions_per_axis = (4, 5)
	generate_function_information(options.input_file, n, no_divisions_per_axis)
	print parse_function_information(options.input_file, no_divisions_per_axis)

if __name__ == '__main__':
	main()