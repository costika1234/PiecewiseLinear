import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import sympy as sp
from sympy import poly

def get_polynomial_vars(n):
    return sp.symbols(['x_' + str(i + 1)for i in range(n)])


def get_coefs_exponents_list(n):
    result = []
    for index in range(10):
        coef = np.random.randint(-40, 40, 1).tolist()
        result.append(tuple([coef[0]] + np.random.randint(0, 5, n).tolist()))

    print result
    return result


def build_polynomial(coefs_exponents_list):
    # Construct an 'n'-dimensional polynomial from the given coefficients and exponents list, 
    # given in the form: [(coef_1, exp_1, exp_2, ..., exp_n), ...].
    poly_terms = []
    poly_vars = get_polynomial_vars(len(coefs_exponents_list[0][1:]))

    for coef_exponents in coefs_exponents_list:
        coef = coef_exponents[0]
        poly_term = []
        for index, exponent in enumerate(coef_exponents[1:]):
            poly_term.append(str(poly_vars[index]) + '**' + str(exponent))
        
        poly_terms.append(str(coef) + ' * ' + " * ".join(poly_term))

    return poly(' + '.join(poly_terms), gens=poly_vars)


def fun(polynomial, x, y):
  return float(polynomial.eval((x, y)))

polynomial = build_polynomial(get_coefs_exponents_list(2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)

zs = np.array([fun(polynomial, x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()