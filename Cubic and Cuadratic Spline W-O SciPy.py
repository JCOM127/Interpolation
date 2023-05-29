import numpy as np
import matplotlib.pyplot as plt

def spline_interpolate(x, y, d, eval_point=None):
    """
    Perform cubic or quadratic spline interpolation on the given data points.

    Parameters:
    - x: 1D numpy array, the x-coordinates of the data points.
    - y: 1D numpy array, the y-coordinates of the data points.
    - d: int, the degree of the polynomial for interpolation (2 for quadratic, 3 for cubic).
    - eval_point: float, the point to evaluate the spline polynomial.

    Returns:
    - coeffs: 2D numpy array, the coefficients of the spline polynomials.

    The spline interpolation is done by constructing piecewise polynomials between consecutive data points.

    The coefficients are stored in a 2D numpy array where each row corresponds to a polynomial segment.
    The columns represent the coefficients of the polynomial in descending order of degree.

    For cubic splines (d=3), the columns represent the coefficients [a, b, c, d] of the polynomial:
        p(x) = a + b(x - x_i) + c(x - x_i)^2 + d(x - x_i)^3

    For quadratic splines (d=2), the columns represent the coefficients [a, b, c] of the polynomial:
        p(x) = a + b(x - x_i) + c(x - x_i)^2

    The function also plots the graph of the interpolated polynomial along with the data points.

    Raises:
    - ValueError: If the degree `d` is not 2 or 3.
    """

    if d != 2 and d != 3:
        raise ValueError("Degree of polynomial must be 2 or 3.")

    n = len(x)
    h = np.diff(x)
    b = np.diff(y) / h
    u = np.zeros(n)
    v = np.zeros(n)
    z = np.zeros(n)

    # Set endpoint conditions
    if d == 2:
        u[0] = 0
        v[0] = 2
        u[-1] = 2
        v[-1] = 0
    elif d == 3:
        u[0] = 1
        v[0] = 0
        u[-1] = 0
        v[-1] = 1

    # Calculate intermediate values
    for i in range(1, n-1):
        u[i] = h[i-1] / (h[i-1] + h[i])
        v[i] = 1 - u[i]
        z[i] = 6 * (b[i] - b[i-1]) / (h[i-1] + h[i])

    # Set up the tridiagonal system
    A = np.zeros((n, n))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, n-1):
        A[i, i-1] = u[i]
        A[i, i] = 2
        A[i, i+1] = v[i]

    # Solve the tridiagonal system
    z = np.linalg.solve(A, z)

    # Calculate the coefficients for each segment
    c = np.zeros((n-1, d+1))
    for i in range(n-1):
        c[i, 0] = y[i]
        c[i, 1] = b[i] - h[i] * (2*z[i]+z[i+1]) / 6
        c[i, 2] = z[i] / 2
        if d == 3:
            c[i, 3] = (z[i+1] - z[i]) / (6 * h[i])

    
    fig, ax = plt.subplots()
      
    # Plot the results
    ax.scatter(x, y, marker='o', label='data', color='red')
    xx = np.linspace(x[0], x[-1], 100)
    yy = np.zeros_like(xx)
    for i in range(n-1):
        mask = (xx >= x[i]) & (xx <= x[i+1])
        x_seg = xx[mask]
        y_seg = np.polyval(np.flip(c[i, :d+1]), x_seg - x[i])
        yy[mask] = y_seg
    ax.plot(xx, yy,  label=f'polynomial of degree {d}')

    if eval_point is not None:
        for i in range(n-1):
            if eval_point >= x[i] and eval_point <= x[i+1]:
                eval_result = np.polyval(np.flip(c[i, :d+1]), eval_point - x[i])
                ax.scatter(eval_point, eval_result, marker='o', color='blue', label='Eval Point')
                print("Coefficients (ordered):", *np.flip(c[i, :d+1]), sep=", ")
                print(f"Result at eval_point ({eval_point}):", eval_result)
                break
    ax.set_title('Cubic Spline Interpolation')
    ax.legend(loc='best')
    ax.set_xlabel("Altitude (ft)")
    ax.set_ylabel("Fuel in (L)")
    ax.grid()
    plt.show()

    return c


x = np.array([5000, 10000, 15000, 20000, 25000])
y = np.array([2000, 1500, 1200, 1000, 900])
eval_point = 17000
d = 3
coeffs = spline_interpolate(x, y, d, eval_point)
