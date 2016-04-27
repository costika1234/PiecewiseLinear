#/usr/local/bin/python
 
from optparse import OptionParser

INPUT_FILE = "input.txt"

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

def parse_input_file():
	# Parse the input file. Assuming maps [a, b]^n -> R, we define the following
	# format:
	# 
	#   n = domain size
	#   
	#   followed by:
	#   
	#   list of 'n' rows, each partioning the i-th coordinate axis, i = 0, n - 1
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

def command_line_arguments():
    usage = """

    consistency.py:

   	Purpose: this script will check if the pair of step functions (f, g), where
   	f : U -> IR and g : U -> IR^n (the function and derivative information, 
   	respectively) are consistent, i.e. if there exists a third map 'h' that is 
   	approximated by the first component of the pair ('f') and whose derivative
   	is approximated by the second compenent of the pair ('g'). Furthermore, if
   	such a map exists, the script will return the least and greatest such maps,
   	that have the added property of being piece-wise linear.
    
    """

    parser = OptionParser(usage=usage)

    parser.add_option("", "--domain-size", dest="domain_size",
                      help="Specifies the size of the function domain.")

    return parser.parse_args()

def main():
	(options, args) = command_line_arguments()
	
	parse_input_file()

if __name__ == '__main__':
	main()