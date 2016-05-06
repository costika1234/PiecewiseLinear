#/usr/local/bin/python
 
from optparse import OptionParser

import re
import numpy as np

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound' 
FLOAT_NUMBER_REGEX  = r"[-+]?[0-9]*\.?[0-9]+"
FUNCTION_INFO_DTYPE = ('float64, float64')

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


def get_zipped_list(no_elements):
	# Returns a list of the form: [(a, b), ...] to help generating dummy data.
	l = np.arange(no_elements)
	u = l[::-1]
	return zip(l, u)


def build_tuples_regex_for_dimension(d):
	return r"\((?P<" + LOWER_BOUND + r"_" + str(d) + r">" + FLOAT_NUMBER_REGEX + r"),\s*" + \
             r"(?P<" + UPPER_BOUND + r"_" + str(d) + r">" + FLOAT_NUMBER_REGEX + r")\)"	


def build_tuples_regex(n, is_function_info):
	# Constructs a regex to match either (x, y) - for function information, or
	# ((a, b), (c, d), ...) - for derivative information.
	if is_function_info:
		return build_tuples_regex_for_dimension(1)
	
	return r"\(" + r",\s*".join([build_tuples_regex_for_dimension(d + 1) for d in range(n)]) + r"\)"


def generate_function_info(input_file, n, no_divisions_per_axis):
	# Generate some lower and upper bounds sequences and initialize the n-dimensional array
	# consisting of one pair for each entry.
	no_elements = np.prod(no_divisions_per_axis) 
	nd_array = np.array(get_zipped_list(no_elements), dtype=FUNCTION_INFO_DTYPE) \
		.reshape(no_divisions_per_axis)

	# Write contents to file.
	with file(input_file, 'w+') as f:
		f.write('# Array shape: {0}\n'.format(nd_array.shape))
		traverse_nd_array(nd_array, f, n)

	return nd_array


def generate_derivative_info(input_file, n, no_divisions_per_axis): 
	# Generate some lower and upper bounds sequences and initialize the n-dimensional array
	# consisting of 'n' tuples for each entry (corresponding to each partial derivative along
	# the 'n' axis of the domain).
	no_elements = np.prod(no_divisions_per_axis)
	zipped = get_zipped_list(no_elements)
	flat_derivative_info = zip(*[zipped for _ in range(n)])

	dt = get_dtype(n, False)
	nd_array = np.array(flat_derivative_info, dtype=dt).reshape(no_divisions_per_axis)

	# Write contents to file.
	with file(input_file, 'w+') as f:
	    f.write('# Array shape: {0}\n'.format(nd_array.shape))
	    traverse_nd_array(nd_array, f, n)

	return nd_array


def traverse_nd_array(nd_array, f, depth):
	# Function which recursively traverses an n-dimenionsal array and saves the array to file.
	if depth == 2:
		np.savetxt(f, nd_array, fmt='%s')
	else:
		for sub_nd_array in nd_array:
			traverse_nd_array(sub_nd_array, f, depth - 1)

	f.write('# End of %d depth \n' % depth)


def build_tuple_match(n, match, is_function_info):
	if is_function_info:
		return tuple([match.group(LOWER_BOUND + "_1"), match.group(UPPER_BOUND + "_1")])

	return tuple([(match.group(LOWER_BOUND + "_%d" % (d + 1)), \
		           match.group(UPPER_BOUND + "_%d" % (d + 1))) for d in range(n)])


def get_dtype(n, is_function_info):
	if is_function_info:
		return ('float64, float64')

	tuple_dtype = [('lower_bound', 'float64'), ('upper_bound', 'float64')]
	return [(str(dim + 1), tuple_dtype) for dim in range(n)]


def parse_tuples_info(input_file, n, no_divisions_per_axis, is_function_info):
	flat_nd_list = []
	regex = build_tuples_regex(n, is_function_info)

	with open(input_file) as f:
		for line in f:
			# Ignore possible comments in the input lines.
			if line.startswith('#'):
				continue

			# Append the pairs of lower and upper bounds to the flat list.
			for match in re.finditer(regex, line):
				flat_nd_list.append(build_tuple_match(n, match, is_function_info))

	# Finally, convert to the shape of an n-dimensional array from the given divisions.
	return np.array(flat_nd_list, \
		            dtype=get_dtype(n, is_function_info)).reshape(no_divisions_per_axis) 


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
    parser.add_option("", "--dimension", dest="dimension",
                      help="Specifies the dimension of the function domain.")
    

    return parser.parse_args()


def main():
	(options, args) = command_line_arguments()

	n = int(options.dimension)
	no_divisions_per_axis = (4,) * n
	
	print "FUNCTION INFORMATION"
	a = generate_function_info(options.input_file, n, no_divisions_per_axis)
	b = parse_tuples_info(options.input_file, n, no_divisions_per_axis, True)
	print a
	print b
	print a == b

	print "\nDERIVATIVE INFORMATION"
	c = generate_derivative_info(options.input_file, n, no_divisions_per_axis)
	d = parse_tuples_info(options.input_file, n, no_divisions_per_axis, False)
	print c
	print d
	print c == d

if __name__ == '__main__':
	main()