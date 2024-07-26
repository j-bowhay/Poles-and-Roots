# %%
import numpy as np
import matplotlib.pyplot as plt

from poles_roots.global_zero_pole import find_zeros_poles
from poles_roots.plotting import phase_plot, plot_poles_zeros


# %%
def compute_true_poles(n_max, d=1, sign=-1):
    n = np.arange(n_max)
    return sign * np.emath.sqrt(1 - (2 * n + 1) ** 2 * np.pi**2 / (4 * d**2))


def f_full(z, d, y):
    return np.sin(np.emath.sqrt(1 - z**2) * (np.abs(y) - d)) / (
        2 * np.emath.sqrt(1 - z**2) * np.cos(np.emath.sqrt(1 - z**2) * d)
    )


# %%
d = 1 - 10j
n = 70

poles_p = compute_true_poles(n, d=d, sign=1)

fig, ax = plt.subplots()

phase_plot(
    lambda z: f_full(z, d, 0),
    ax,
    domain=[
        poles_p.real.min(),
        poles_p.real.max(),
        poles_p.imag.min(),
        poles_p.imag.max(),
    ],
)
ax.plot(poles_p.real, poles_p.imag, ".")
plt.show()


# %%
def f(z, d):
    return (1 / 2) / (np.sqrt(1 - z**2) * np.cos(d * np.sqrt(1 - z**2)))


def f_prime(z, d):
    return (
        -1
        / 2
        * z
        * (
            d * np.sqrt(1 - z**2) * np.sin(d * np.sqrt(1 - z**2))
            - np.cos(d * np.sqrt(1 - z**2))
        )
        / ((1 - z**2) ** (3 / 2) * np.cos(d * np.sqrt(1 - z**2)) ** 2)
    )


# %%
fig, ax = plt.subplots()
points = np.asarray([0.5 + 0.1j, 0.5 - 2.5j, 25 - 2.5j, 25 + 0.1j])

phase_plot(
    lambda z: f_prime(z, d) / f(z, d),
    ax,
    domain=[points.real.min(), points.real.max(), points.imag.min(), points.imag.max()],
)
ax.plot(poles_p.real, poles_p.imag, ".")


ax.plot(points.real, points.imag, "-c.", markersize=20)
ax.set_ylim(ax.yaxis.get_data_interval())
ax.set_xlim(ax.xaxis.get_data_interval())

# %%
res = find_zeros_poles(
    lambda z: f(z, d),
    lambda z: f_prime(z, d),
    points,
    100,
    arg_principal_threshold=10,
    cross_ref=False,
)

# %%
fig, ax = plt.subplots()
plot_poles_zeros(res, ax, expected_poles=poles_p)
ax.triplot(res.points[:, 0], res.points[:, 1], res.simplices)

# %%
errors = np.min(np.abs(np.subtract.outer(poles_p, res.poles)), axis=1)
fig, ax = plt.subplots()
ax.hist(np.log10(errors + 5e-17))

# %%
res.residues

# %%
errors


# %%
