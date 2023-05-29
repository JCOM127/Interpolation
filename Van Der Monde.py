import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['backend'] = 'TkAgg'

def vandermonde(x, y, eval_point):
    """
    Generate the Van Der Monde interpolation of the input values and plot the initial values
    and the resulting polynomial.

    Parameters:
    x (array): x values of the data points.
    y (array): y values of the data points.
    eval_point (float): Point to evaluate the polynomial.

    Returns:
    None.

    The function generates the Van Der Monde matrix based on the input x values and computes
    the coefficients of the polynomial using the linear equation system solved through
    numpy.linalg.solve. A linspace is then created based on the minimum and maximum values of x,
    including the evaluation point, and y values are computed using the polynomial coefficients.
    Finally, the initial values and the polynomial are plotted using matplotlib.pyplot.scatter
    and matplotlib.pyplot.plot functions. The function also evaluates the provided point and
    displays it on the graph as a scatter point.
    """
    A = np.vander(x, increasing=False)
    coeff = np.linalg.solve(A, y)
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_plot = np.polyval(coeff, x_plot)
    eval_value = np.polyval(coeff, eval_point)
    print("Coefficients:", coeff)
    print(f"Value at eval_point ({eval_point}):", eval_value)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color='red', label='Initial Values')
    ax.plot(x_plot, y_plot, color='blue', label='Polynomial')
    ax.scatter(eval_point, eval_value, color='green', label='Evaluation Point')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Van der Monde Interpolation")
    ax.legend()
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 100, y.max() + 100)
    ax.grid()
    plt.show()

x = np.array([5000, 10000, 15000, 20000, 25000])
y = np.array([2000, 1500, 1200, 1000, 900])
eval_point = 17000
vandermonde(x, y, eval_point)
