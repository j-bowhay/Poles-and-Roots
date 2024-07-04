import numpy as np

h = np.sqrt(np.finfo(np.float64).eps)


def derivative(f, z, /):
    """Forwards difference."""
    return (f(z + h) - f(z)) / h


def complex_derivitive(f, z, N):
    k = np.arange(N)
    zk = np.exp(2 * np.pi * 1j * k / N) + z
    fk = f(zk)
    return np.mean(zk * fk / (z - zk) ** 2)
