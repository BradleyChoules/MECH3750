"""
MECH3750: Engineering Analysis II

PBL 01: A Refresher in Python, NumPy and matplotlib

Temperature distribution in a hot wire with fixed end temperatures, solving the
system of linear equations (with Dirichlet boundary conditions),

theta_(i-1) + sigma*theta_i + theta_(i+1) ~= 0,

where sigma = -2 - beta^2*dx^2 and,

beta^2 = (h*P)/(k*A).

In SI units, the inputs and parameters to these equations are as follows.

L [m]:      The length of the wire
d [m]:      The diameter of the wire
P [m]:      The perimeter of the wire
A [m^2]:    The cross-sectional area of the wire
k [W/mK]:   The thermal conductivity of the wire
h [W/m^2K]: The heat transfer coefficient of the wire
T0 [K]:     The temperature at the left end (x = 0) of the wire
TL [K]:     The temperature at the right end (x = L) of the wire
Ta [K]:     The ambient fluid temperature
"""

__author__ = 'Christopher Leonardi'
__version__ = '0.0.1'
__date__ = '01/05/2021'

import numpy as np
import matplotlib.pyplot as plt

def hot_wire_01(L, n, d, k, h, T0, TL, Ta):
    """
    Calculation of temperature distribution in a hot wire anemometer
    n [-]: The number of calculation points, endpoints inclusive
    """
    x_array = np.linspace(0., L, n)     # 1D array of discretisation points
    theta_n = np.zeros_like(x_array)    # 1D array of excess temperatures
    theta_a = np.zeros_like(x_array)    # 1D array of analytical solution

    # Calculate some parameters needed in the analysis
    dx = x_array[1] - x_array[0]    # The discretisation spacing [m]
    P = np.pi*d                     # Perimeter of the wire [m]
    A = (np.pi*d*d)/4.0             # Cross-sectional area of the wire [m^2]
    beta2 = (h*P)/(k*A)             # Lumped heat transfer coefficient [1/m^2]
    beta = np.sqrt(beta2)
    theta0 = T0 - Ta                # Excess temperature at left end (x = 0)
    thetaL = TL - Ta                # Excess temperature at right end (x = L)

    # Create the coefficient matrix for the system of equations
    # Note that the adjacent diagonals require one less input term
    a = np.ones(n-1)
    b = -np.ones(n)*(2.0+(beta*dx)**2.0)
    c = a
    M = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)

    # Set the top and bottom rows to zero to prepare for boundary conditions
    M[0,:] = 0.0
    M[n-1,:] = 0.0

    # Dirichlet boundary condition at left end (x = 0)
    M[0,0] = 1.0

    # Dirichlet boundary condition at right end (x = L)
    M[n-1,n-1] = 1.0

    # Create the RHS array for our system of equations
    b = np.zeros(n)
    b[0] = theta0
    b[n-1] = thetaL

    # Solve the system of linear equations
    theta_n = np.linalg.solve(M, b)

    # Calculate the analytical solution
    theta_a = ((thetaL*np.sinh(beta*x_array))+ \
               (theta0*np.sinh(beta*(L-x_array))))/np.sinh(beta*L)

    # Plot the numerical and analytical solutions
    fig,ax = plt.subplots()
    ax.plot(x_array,theta_n+Ta,x_array,theta_a+Ta,'go')
    fig.suptitle('Temperature Distribution in a Hot Wire')
    ax.set(xlabel='Wire Coordinate (m)', ylabel='Temperature (K)')
    ax.legend(['Numerical','Analytical'])
    ax.grid()
    plt.show()

    return x_array, theta_n, theta_a
    


