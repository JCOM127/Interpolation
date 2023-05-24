import numpy as np
import matplotlib.pyplot as plt

def lagrange(x, y, t):
    """
    Find the Lagrange polynomial through the points (x, y) and return its value at t.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    - t: float, the value at which to evaluate the Lagrange polynomial.

    Returns:
    - p: float, the value of the Lagrange polynomial at t.
    """

    # Check that the input arrays have the same length
    if len(x) != len(y):
        raise ValueError("The arrays x and y must have the same length.")

    # Initialize the polynomial
    p = 0

    # Loop over the points
    for i in range(len(x)):
        # Get the current point
        xi, yi = x[i], y[i]

        # Compute the Lagrange basis polynomial
        basis = np.prod([(t - x[j]) / (xi - x[j]) for j in range(len(x)) if i != j])

        # Add the term to the polynomial
        p += yi * basis

    return p


def plot_lagrange(x, y, t):
    """
    Plot the data points and the Lagrange polynomial.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    - t: float, the value at which to evaluate the Lagrange polynomial.
    """

    # Plot the data points
    plt.scatter(x, y, color='red', label='Data Points')

    # Generate points on the Lagrange polynomial for plotting
    plot_x = np.linspace(np.min(x), np.max(x), 100)
    plot_y = np.array([lagrange(x, y, t) for t in plot_x])

    # Plot the Lagrange polynomial
    plt.plot(plot_x, plot_y, color='blue', label='Lagrange Polynomial')

    # Plot the evaluated point
    plt.scatter(t, lagrange(x, y, t), color='green', label='Evaluated Point')

    # Set plot labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Lagrange Polynomial')

    # Add a legend
    plt.legend()
    
    #Add Grid
    plt.grid()

    # Show the plot
    plt.show()


# Example usage
x = np.array([100, 200, 500, 900])
y = np.array([8, 15, 25, 28])
t = 400

plot_lagrange(x, y, t)
