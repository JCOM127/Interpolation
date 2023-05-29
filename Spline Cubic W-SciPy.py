import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def spline_interpolation(x, y, d):
    """
    Perform cubic spline interpolation on the given data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    - d: int, the degree of the spline polynomial. Must be 1, 2, or 3.

    Returns:
    - coeffs: 2D numpy array, the coefficients of the spline polynomial.

    Raises:
    - ValueError: If the degree of the polynomial is not 1, 2, or 3.
    """

    cs = CubicSpline(x, y, bc_type='natural')
    if d == 1:
        coeffs = cs.c[:, np.newaxis]
    elif d == 2:
        coeffs = np.hstack([cs.c[:len(x)], np.zeros((len(x), 1))])
    elif d == 3:
        coeffs = cs.c
    else:
        raise ValueError("Degree of polynomial must be 1, 2, or 3")

    # Plot the graph
    plt.scatter(x, y, marker='o', label='data', color='red')
    xs = np.linspace(x[0], x[-1], 100)
    plt.plot(xs, cs(xs), label=f'polynomial of degree {d}')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    return coeffs


# Define the x and y data points
x = np.array([100, 200, 500, 900]) 
y = np.array([8, 15, 25, 28]) 

# Define the degree of the polynomial
d = 3

# Compute the spline polynomial
p = spline_interpolation(x, y, d)
