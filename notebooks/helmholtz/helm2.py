# %%
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("/home/jakeb/development/poles-and-roots/notebooks/science.mplstyle")

from poles_roots.global_zero_pole import find_zeros_poles

# %%
alpha = np.pi / 8
gamma = 10
d = gamma * np.exp(-1j * alpha)
k = 1.75
y_prime = 2


# %%
def sol_branch_1(n, d, gamma, alpha, k):
    return (
        -(1 / d) * np.log(2 * (n * np.pi + alpha) / (gamma * np.sqrt(k**2 - 1)))
        - (n * np.pi + alpha) * 1j / d
    )


def sol_branch_2(n, alpha, k, y_prime):
    return (1 / y_prime) * np.log(
        np.pi * (2 * n + 1) / (y_prime * np.sqrt(k**2 - 1))
    ) - (2 * n + alpha) * np.pi * 1j / (2 * y_prime)


# %%


def N(z):
    return np.sqrt(1 - z**2) * np.sin(y_prime * np.sqrt(k**2 - z**2)) * np.cos(
        d * np.sqrt(1 - z**2)
    ) + np.sqrt(k**2 - z**2) * np.sin(d * np.sqrt(1 - z**2)) * np.cos(
        y_prime * np.sqrt(k**2 - z**2)
    )


def N_prime(z):
    return (
        d * z * np.sin(d * np.sqrt(1 - z**2)) * np.sin(y_prime * np.sqrt(k**2 - z**2))
        - d
        * z
        * np.sqrt(k**2 - z**2)
        * np.cos(d * np.sqrt(1 - z**2))
        * np.cos(y_prime * np.sqrt(k**2 - z**2))
        / np.sqrt(1 - z**2)
        - z
        * y_prime
        * np.sqrt(1 - z**2)
        * np.cos(d * np.sqrt(1 - z**2))
        * np.cos(y_prime * np.sqrt(k**2 - z**2))
        / np.sqrt(k**2 - z**2)
        + z
        * y_prime
        * np.sin(d * np.sqrt(1 - z**2))
        * np.sin(y_prime * np.sqrt(k**2 - z**2))
        - z
        * np.sin(d * np.sqrt(1 - z**2))
        * np.cos(y_prime * np.sqrt(k**2 - z**2))
        / np.sqrt(k**2 - z**2)
        - z
        * np.sin(y_prime * np.sqrt(k**2 - z**2))
        * np.cos(d * np.sqrt(1 - z**2))
        / np.sqrt(1 - z**2)
    )


# %%
res = find_zeros_poles(
    N,
    N_prime,
    initial_points=[0 + 0.01j, 0 - 20j, 3 - 20j, 3 + 0.01j],
    num_sample_points=300,
    arg_principal_threshold=10,
    cross_ref=False,
)
# %%
asym1 = [sol_branch_1(i, d, gamma, alpha, k) for i in range(1, 28)]
asym2 = [sol_branch_2(i, alpha, k, y_prime) for i in range(1, 14)]
fig, ax = plt.subplots()
ax.plot(np.real(asym1), np.imag(asym1), ".", label="Asymptotic Solution Branch 1")
ax.plot(np.real(asym2), np.imag(asym2), ".", label="Asymptotic Solution Branch 2")
ax.plot(res.roots.real, res.roots.imag, "x", label="Computed Zeros")
ax.set_ylabel(r"$\Im(z)$")
ax.set_xlabel(r"$\Re(z)$")
ax.set_xlim([0, 3])
ax.set_ylim([-20, 0])
ax.legend(
    loc="lower center", bbox_to_anchor=(0.4, -0.7), fancybox=True, shadow=True, ncol=1
)
plt.savefig("../figures/helmholtz2_pole_location.pdf")

# %%
from poles_roots.plotting import phase_plot

fig, ax = plt.subplots()
phase_plot(N, ax, domain=[0, 3, -20, 0])
# %%
fig, ax = plt.subplots()
phase_plot(lambda z: N_prime(z) / N(z), ax, domain=[0, 3, -20, 0])
ax.plot(res.roots.real, res.roots.imag, "x", label="Computed Zeros")
# %%
