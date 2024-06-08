import numpy as np

h = np.sqrt(np.finfo(np.float64).eps)


def derivative(f, z, /):
    """Forwards difference."""
    return (f(z + h) - f(z)) / h
