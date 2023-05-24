import matplotlib.pyplot as plt
import numpy as np


def NDD(x, y):
    """
    Calculate the Newton divided differences for the given data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.

    Returns:
    - p: 1D numpy array, the coefficients of the Newton interpolating polynomial.
    """

    n = len(x)
    # Construct table and load x-y pairs in first columns
    A = np.zeros((n, n+1))
    A[:, 0] = x[:]
    A[:, 1] = y[:]
    # Fill in Divided differences
    for j in range(2, n+1):
        for i in range(j-1, n):
            A[i, j] = (A[i, j-1] - A[i-1, j-1]) / (A[i, 0] - A[i-j+1, 0])
    # Copy diagonal elements into array for returning
    p = np.zeros(n)
    for k in range(0, n):
        p[k] = A[k, k+1]
    return p


def poly(t, x, p):
    """
    Evaluates the Newton interpolating polynomial at a given t using the coefficients and x-values.

    Parameters:
    - t: float, the value at which to evaluate the polynomial.
    - x: 1D numpy array, the x-coordinates of the data points.
    - p: 1D numpy array, the coefficients of the Newton interpolating polynomial.

    Returns:
    - out: float, the value of the polynomial at t.
    """

    n = len(x)
    out = p[n-1]
    for i in range(n-2, -1, -1):
        out = out * (t - x[i]) + p[i]
    return out


def plot_interpolation(x, y):
    """
    Plot the Newton interpolating polynomial and data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    """

    # Obtain polynomial coefficients using NDD function
    coefficients = NDD(x, y)

    # Evaluate polynomial at tval
    tval = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)
    yval = poly(tval, x, coefficients)

    # Plot the polynomial and data points
    plt.plot(tval, yval, color='green', linestyle='-', label='Polynomial')
    plt.scatter(x, y, color='blue', marker='o', label='Data Points')

    # Annotate the graph
    plt.title('Interpolation')
    plt.xlabel('x values')
    plt.ylabel('y values')
    plt.legend(loc='best')
    plt.grid()

    # Show the plot
    plt.show()


# Data points
xpt = np.array([100, 200, 500, 900])
ypt = np.array([8, 15, 25, 28])

# Plot the interpolation
plot_interpolation(xpt, ypt)

# Evaluate polynomial at x = 300
x = 300
coefficients = NDD(xpt, ypt)
result = poly(x, xpt, coefficients)
print("Result at x =", x, "is", result)
