import numpy as np
import matplotlib.pyplot as plt



def vandermonde(x, y):
    
    """
    Generate the Van Der Monde interpolation of the input values and plot the initial values
    and the resulting polynomial.

    Parameters:
    x (array): x values of the data points.
    y (array): y values of the data points.

    Returns:
    None.

    The function generates the Van Der Monde matrix based on the input x values and computes
    the coefficients of the polynomial using the linear equation system solved through
    numpy.linalg.solve. A linspace is then created based on the minimum and maximum values of x,
    and y values are computed using the polynomial coefficients. Finally, the initial values and
    the polynomial are plotted using matplotlib.pyplot.scatter and matplotlib.pyplot.plot functions.
    """
    
    
    
    A=np.vander(x, increasing=False) #Order of the powers of the columns. If True, the powers increase from left to right, if False (the default) they are reversed.
    coeff= np.linalg.solve(A, y) #Solve for coefficients
    x_plot = np.linspace(x.min()-10, x.max()+10) #Create linspace within the min and max of x,y
    y_plot = np.polyval(coeff, x_plot) #Create y values for the coefficients/polynomial 
    print(coeff)
    print(A)
    fig, ax = plt.subplots()
    
    ax.scatter(x, y, color='red', label='Initial Values')
    ax.plot(x_plot, y_plot, color='blue', label='Polynomial')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Van der Monde Interpolation")
    ax.legend()
    ax.set_xlim(x.min()-10, x.max()+10)  # Adjusting the x axis limits
    ax.set_ylim(y.min()-10, y.max()+10)  # Adjusting the y axis limits
    ax.grid()
    plt.show()
    
    
    
x = np.array([3, 3.7, 4.4])
y = np.array([6, 10, 15])

vandermonde(x,y)