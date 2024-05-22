import numpy as np

h = np.finfo(np.float64).eps

def derivative(f, z):
    return (f(z + h) - f(z))/h