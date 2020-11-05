A collection of scripts used to check whether a pair of step functions `(f, g)`, where `f : U -> IR` and `g : U -> IR^n` (the function and derivative information, respectively) are consistent, i.e. if there exists a third map `h` that is approximated by the first component of the pair (`f`) and whose derivative is approximated by the second component of the pair (`g`). Furthermore, if such a map exists, the program returns the least and greatest such maps with the added property of being piece-wise linear.

The following libraries are required: `cvxopt`, `sympy`, `numpy`, `matplotlib`.
