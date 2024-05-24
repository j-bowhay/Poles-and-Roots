import warnings
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.linalg
import scipy.sparse


@dataclass
class AAAResult:
    pol: np.ndarray
    res: np.ndarray
    zer: np.ndarray
    zj: np.ndarray
    fj: np.ndarray
    wj: np.ndarray
    errvec: np.ndarray

    def __call__(self, zz) -> np.ndarray:
        # evaluate rational function in barycentric form.
        zz = np.asarray(zz)
        zv = np.ravel(zz)

        # Cauchy matrix
        with np.errstate(invalid="ignore", divide="ignore"):
            CC = 1 / np.subtract.outer(zv, self.zj)
        # Vector of values
        with np.errstate(invalid="ignore"):
            r = CC @ (self.wj * self.fj) / (CC @ self.wj)

        # Deal with input inf: r(inf) = lim r(zz) = sum(w.*f) / sum(w):
        r[np.isinf(zv)] = np.sum(self.wj * self.fj) / np.sum(self.wj)

        # Deal with NaN:
        ii = np.nonzero(np.isnan(r))[0]
        for jj in ii:
            if np.isnan(zv[jj]) or not np.any(zv[jj] == self.zj):
                # r(NaN) = NaN is fine.
                # The second case may happen if r(zv(ii)) = 0/0 at some point.
                pass
            else:
                # Clean up values NaN = inf/inf at support points.
                # Find the corresponding node and set entry to correct value:
                r[jj] = self.fj[zv[jj] == self.zj]

        return np.reshape(r, zz.shape)


def _AAA_iv(F, Z, mmax):
    # Deal with Z and F
    Z = np.ravel(Z)

    # Function values:
    # Work with column vector and check that it has correct length.
    F = np.ravel(F)
    if F.size != Z.size:
        raise ValueError("Inputs `F` and `Z` must have the same length.")

    return F, Z, mmax


def AAA(F, Z, *, tol=1e-13, mmax=100, cleanup=True, cleanup_tol=1e-13) -> AAAResult:
    F, Z, mmax = _AAA_iv(F, Z, mmax)

    # Currently we don't handle `F` being callable

    # Remove infinite or NaN function values and repeated entries:
    to_keep = (np.isfinite(F)) & (~np.isnan(F))
    F = F[to_keep]
    Z = Z[to_keep]
    _, uni = np.unique(Z, return_index=True)
    Z = Z[uni]
    F = F[uni]

    # Initialization for AAA iteration:
    M = np.size(Z)
    # absolute tolerance
    abstol = tol * np.linalg.norm(F, ord=np.inf)
    J = np.arange(M)
    zj = np.empty(mmax, dtype=np.complex128)
    fj = np.empty(mmax, dtype=np.complex128)
    # Cauchy matrix
    C = np.empty((M, mmax), dtype=np.complex128)
    # Loewner matrix
    A = np.empty((M, mmax), dtype=np.complex128)
    errvec = np.empty(mmax, dtype=np.complex128)
    R = np.mean(F) * np.ones_like(J)

    # AAA iteration
    for m in range(mmax):
        # Introduce next support point
        # Select next support point
        jj = np.argmax(np.abs(F[J] - R[J]))
        # Update support points
        zj[m] = Z[J[jj]]
        # Update data values
        fj[m] = F[J[jj]]
        # Next column of Cauchy matrix
        with np.errstate(divide="ignore", invalid="ignore"):
            C[:, m] = 1 / (Z - Z[J[jj]])
        # Update index vector
        # TODO: switch to boolean mask once happy with everything else
        J = np.delete(J, jj)
        # Update Loewner matrix
        with np.errstate(invalid="ignore"):
            A[:, m] = (F - fj[m]) * C[:, m]

        # Compute weights:
        # The usual tall-skinny case
        if J.size >= m + 1:
            # Reduced SVD
            _, s, V = scipy.linalg.svd(A[J, : m + 1], full_matrices=False)
            V = V.conj().T
            # Treat case of multiple min sing val
            mm = np.nonzero(s == np.min(s))[0]
            nm = mm.size
            # Aim for non-sparse wt vector
            wj = V[:, mm] @ np.ones(nm) / np.sqrt(nm)
        elif J.size >= 1:
            # Fewer rows than columns
            V = scipy.linalg.null_space(A[J, : m + 1])
            nm = V.shape[-1]
            # Aim for non-sparse wt vector
            wj = V @ np.ones(nm) / np.sqrt(nm)
        else:
            # No rows at all (needed for Octave) DO WE NEED THIS
            wj = np.ones(m + 1) / np.sqrt(m + 1)

        # Compute rational approximant:
        # Omit columns with wj = 0
        i0 = np.nonzero(wj)[0]
        with np.errstate(invalid="ignore"):
            # Numerator
            N = C[:, : m + 1][:, i0] @ (wj[i0] * fj[: m + 1][i0])
            # Denominator
            D = C[:, : m + 1][:, i0] @ wj[i0]
        # Interpolate at supp pts with wj~=0
        D_inf = np.isinf(D) | np.isnan(D)
        D[D_inf] = 1
        N[D_inf] = F[D_inf]
        R = N / D

        # Check if converged:
        max_err = np.linalg.norm(F - R, ord=np.inf)
        errvec[m] = max_err
        if max_err <= abstol:
            break

    if m == mmax - 1:
        warnings.warn(f"Failed to converge within {mmax} iterations", stacklevel=2)

    # Trim off unused array allocation
    zj = zj[: m + 1]
    fj = fj[: m + 1]
    C = C[:, : m + 1]
    A = A[:, : m + 1]
    errvec = errvec[: m + 1]

    # Remove support points with zero weight:
    i_non_zero = np.nonzero(wj)[0]
    zj = zj[i_non_zero]
    wj = wj[i_non_zero]
    fj = fj[i_non_zero]

    # Compute poles, residues and zeros:
    pol, res, zer = _prz(zj, fj, wj)

    if cleanup:
        wj, zj, fj, Z, F = _clean_up(pol, res, wj, zj, fj, Z, F, cleanup_tol)
        pol, res, zer = _prz(zj, fj, wj)

    return AAAResult(pol, res, zer, zj, fj, wj, errvec)


def _prz(zj, fj, wj):
    # Compute poles, residues, and zeros of rational fun in barycentric form.

    # Compute poles via generalized eigenvalue problem:
    m = wj.size
    B = np.eye(m + 1)
    B[0, 0] = 0

    E = np.zeros_like(B, dtype=np.complex128)
    E[0, 1:] = wj
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], zj)

    pol = scipy.linalg.eigvals(E, B)
    pol = pol[np.isfinite(pol)]

    # Compute residues via formula for res of quotient of analytic functions:
    N = (1 / (pol[:, np.newaxis] - zj)) @ (fj * wj)
    Ddiff = -((1 / np.subtract.outer(pol, zj)) ** 2) @ wj
    res = N / Ddiff

    # Compute zeros via generalized eigenvalue problem:
    E = np.zeros_like(B, dtype=np.complex128)
    E[0, 1:] = wj * fj
    E[1:, 0] = 1
    np.fill_diagonal(E[1:, 1:], zj)

    zer = scipy.linalg.eigvals(E, B)
    zer = zer[np.isfinite(zer)]

    return pol, res, zer


def _clean_up(pol, res, w, z, f, Z, F, cleanup_tol):
    # Remove spurious pole-zero pairs.

    # Find negligible residues:
    if np.any(F):
        geometric_mean_of_abs_F = np.exp(np.mean(np.log(np.abs(F[np.nonzero(F)]))))
    else:
        geometric_mean_of_abs_F = 0

    Z_distances = np.empty(pol.size, dtype=np.complex128)

    for j, pol_j in enumerate(pol):
        Z_distances[j] = np.min(np.abs(pol_j - Z))

    ii = np.nonzero(np.abs(res) / Z_distances < cleanup_tol * geometric_mean_of_abs_F)[
        0
    ]
    ni = ii.size
    if ni == 0:
        return w, z, f, Z, F
    else:
        warnings.warn(f"{ni} Froissart doublets detected.", stacklevel=3)

    # For each spurious pole find and remove closest support point:
    for j in range(ni):
        azp = np.abs(z - pol[ii[j]])
        jj = np.argmin(azp)

        # Remove support point(s)
        z = np.delete(z, jj)
        f = np.delete(f, jj)

    # Remove support points z from sample set
    for z_jj in z:
        to_keep = Z != z_jj
        F = F[to_keep]
        Z = Z[to_keep]

    m = z.size
    M = Z.size

    # Build Loewner matrix:
    SF = scipy.sparse.spdiags(F, 0, M, M)
    Sf = np.diag(f)
    # Cauchy matrix
    C = 1 / np.subtract.outer(Z, z)
    # Loewner matrix
    A = SF @ C - C @ Sf

    # Solve least-squares problem to obtain weights:
    _, _, V = scipy.linalg.svd(A, full_matrices=False)
    V = V.conj().T
    w = V[:, m - 1]

    return w, z, f, Z, F
