import numpy as np
import matplotlib.pyplot as plt

def spline_interp(x, y):
    """
    Perform linear spline interpolation on the given data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.

    Returns:
    - poly: 2D numpy array, the coefficients of the linear spline polynomial.

    The linear spline is defined by a piecewise linear function where each segment
    between two data points is a linear polynomial.

    The polynomial is represented by two arrays:
    - poly[0]: 1D numpy array, the slopes of the linear segments.
    - poly[1]: 1D numpy array, the y-intercepts of the linear segments.

    The length of poly[0] and poly[1] is one less than the length of x and y.

    The function also plots the graph of the interpolated polynomial along with the data points.
    """

    # Compute the coefficients of the linear polynomial
    m = np.zeros_like(x)
    b = np.zeros_like(x)
    m[:-1] = np.diff(y) / np.diff(x)
    b[:-1] = y[:-1] - m[:-1] * x[:-1]

    # Create the x and y values for the polynomial
    poly_x = np.linspace(x[0], x[-1], 1000)
    poly_y = np.interp(poly_x, x, y)

    # Plot the data and the polynomial
    plt.scatter(x, y, marker='o', label='data', color='red')
    plt.plot(poly_x, poly_y, label='Polynomial')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Return the polynomial
    return np.array((m, b))


x = np.array([100, 200, 500, 900]) 
y = np.array([8, 15, 25, 28]) 

poly = spline_interp(x, y)
