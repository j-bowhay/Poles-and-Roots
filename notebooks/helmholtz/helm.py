# %%
import time

import numpy as np
import scipy
import matplotlib.pyplot as plt

plt.style.use("/home/jakeb/development/poles-and-roots/notebooks/science.mplstyle")
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
d = 10 * np.exp(-1j * np.pi / 4)
n = 100

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
points = np.asarray([0.5 + 0.1j, 0.5 - 22.5j, 22.5 - 22.5j, 22.5 + 0.1j])

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
rng = np.random.default_rng(1)
tic = time.perf_counter()
res = find_zeros_poles(
    lambda z: f(z, d),
    lambda z: f_prime(z, d),
    points,
    1000,
    arg_principal_threshold=10,
    cross_ref=False,
    rng=rng,
)
toc = time.perf_counter()

print(toc - tic)

# %%
fig, ax = plt.subplots()
ax.triplot(res.points[:, 0], res.points[:, 1], res.simplices)
ax.plot(
    np.real(res.poles),
    np.imag(res.poles),
    "rx",
    markersize=5,
    label="Computed Poles",
)
ax.legend(
    loc="lower center", bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=1
)
ax.set_xlabel(r"$\Re(z)$")
ax.set_ylabel(r"$\Im(z)$")
plt.savefig("../figures/helmholtz_pole_location.pdf")

# %%
errors = np.min(np.abs(np.subtract.outer(poles_p, res.poles)), axis=1)
fig, ax = plt.subplots()
ax.hist(np.log10(errors + 5e-17), alpha=0.5)
ax.set_xlabel("Log Error in Pole Location")
ax.set_ylabel("Frequency")
plt.savefig("../figures/helmholtz_pole_error.pdf")

# %%


def residue_anal(x, y, n, d):
    return (
        (-1j / (2 * d))
        * np.cos((2 * n + 1) * np.pi * y / (2 * d))
        * np.exp(-1j * x * pole(n, d))
        / pole(n, d)
    )


def green(x, y):
    return (1j / 2) * scipy.special.hankel2(0, np.sqrt(x**2 + y**2))


def pole(n, d):
    return np.emath.sqrt(1 - ((2 * n + 1) ** 2 * np.pi**2 / (4 * d**2)))


def hankel_approx(x, n, d):
    return (2 / d) * np.exp(-1j * x * pole(n, d)) / pole(n, d)


def hankel_approx2(x, pole):
    return (2 / d) * np.exp(-1j * x * pole) / pole


# %%
d = 10 * np.exp(-1j * np.pi / 4)
fig, ax = plt.subplots()
ax.plot(
    np.abs(
        scipy.special.hankel2(0, 1) - np.cumsum(hankel_approx(1, np.arange(100), d))
    ),
    label="Analytical",
)
ax.plot(
    np.abs(
        scipy.special.hankel2(0, 1) - np.cumsum(hankel_approx2(1, np.sort(res.poles)))
    ),
    label="Computational",
)
ax.set_yscale("log")
ax.set_xlabel(r"\# Terms")
ax.set_ylabel("Error In Approximation")
ax.legend()
plt.savefig("../figures/helmholtz_series_error.pdf")
# %%
