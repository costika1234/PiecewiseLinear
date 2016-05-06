#/usr/local/bin/python
 
from optparse import OptionParser

import re
import numpy as np

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'
FLOAT_NUMBER_REGEX  = r"[-+]?[0-9]*\.?[0-9]+"


def parse_input_file(input_file):
	pass


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


def get_zipped_list(no_elements):
	# Returns a list of the form: [(a, b), ...] to help generating dummy data.
	l = np.arange(no_elements)
	u = l + 1
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

	# Write contents to file.
	file_descriptor.write('# Array shape: {0}\n'.format(nd_array.shape))
	traverse_nd_array(nd_array, file_descriptor, n)


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


def parse_tuples_info(input_file, n, no_points_per_axis, is_function_info):
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

	# Finally, convert to the shape of an n-dimensional array from the given points.
	return np.array(flat_nd_list, \
		            dtype=get_dtype(n, is_function_info)).reshape(no_points_per_axis) 


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
	no_points_per_axis = (5, ) * n
	
	generate_test_file(options.input_file, n, no_points_per_axis)


if __name__ == '__main__':
	main()