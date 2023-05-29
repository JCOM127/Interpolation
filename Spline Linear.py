import numpy as np
import matplotlib.pyplot as plt


def spline_interp(x, y, eval_point=None):
    """
    Perform linear spline interpolation on the given data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    - eval_point: float, the point to evaluate the spline polynomial.

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

    if eval_point is not None:
        eval_result = np.interp(eval_point, x, y)
        plt.scatter(eval_point, eval_result, marker='o', color='green', label='Eval Point')

    plt.legend(loc='best')
    plt.grid()
    plt.show()

    # Print the coefficients (ordered)
    print("Coefficients (ordered):")
    for slope, intercept in zip(m, b):
        print(f"Slope: {slope}, Intercept: {intercept}")

    # Print the evaluation result if eval_point is provided
    if eval_point is not None:
        print(f"Result at eval_point ({eval_point}): {eval_result}")

    # Return the polynomial
    return np.array((m, b))


x = np.array([5000, 10000, 15000, 20000, 25000])
y = np.array([2000, 1500, 1200, 1000, 900])
eval_point = 17000

poly = spline_interp(x, y, eval_point)
