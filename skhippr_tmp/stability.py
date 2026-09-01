from skhippr.stability.KoopmanHillProjection import KoopmanHillProjection
from typing import override
import numpy as np
from scipy.linalg import (
    expm,
    schur,
    lu_factor,
    lu_solve,
    solve_sylvester,
    solve_triangular,
)


class KoopmanHillDAE(KoopmanHillProjection):
    def __init__(self, fourier, tol=0, autonomous=False, tol_drazin=1e-6):
        super().__init__(fourier, tol, autonomous)
        self.tol_drazin = tol_drazin

    @override
    def fundamental_matrix(self, t_over_period, hbm, omega=None, update=True):
        C = self.C_time(t_over_period)
        hill_matrix = hbm.hill_matrix(update=update)

        if omega is None:
            omega = hbm.omega
        else:
            pass
        t = t_over_period * 2 * np.pi / hbm.omega

        if hbm.ode.invertible:
            hill_matrix_inv = np.linalg.solve(hbm.M(), hill_matrix)
            funda_mat = C @ expm(hill_matrix_inv * t) @ self.W
        else:
            funda_mat = (
                C
                @ generalized_exponential(hbm.M(), hill_matrix, t, self.tol_drazin)[0]
                @ self.W
            )

        return funda_mat

    @override
    def error_bound(self, t, a, b):
        raise NotImplementedError("Error bound not applicable for DAEs.")


def drazin(A, tol=0, ax_plot=None, x_value=None):
    """Compute the Drazin inverse of a matrix A using the Schur decomposition.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix.
    tol : float, optional
        Tolerance for determining the rank (default is 0).
    x_value: float, optional
        x value(s) to plot the eigenvalues at, if multiple cases are to be compared in one plot.
        If None (default), the eigenvalues are plotted at their index
    Returns
    -------
    np.ndarray
        The Drazin inverse of the matrix A.
    """
    return drazin_schur_2(A, tol, ax_plot=ax_plot, x_value=x_value)

    if ax_plot is not None:
        eigenvalues = np.diag(T)

        if x_value is None:
            x_vals = np.arange(len(eigenvalues))
        else:
            x_vals = x_value * np.ones_like(eigenvalues)
        # eigenvalues_plot = np.zeros_like(eigenvalues)
        # for k, eigenvalue in enumerate(eigenvalues):
        #     eigenvalues_plot[k] = round_to_significant_digits(eigenvalue, 2)

        # _, idx_unique = np.unique(eigenvalues_plot, return_index=True)
        ax_plot.semilogy(
            x_vals,
            np.abs(eigenvalues),  # [idx_unique]),
            "x",
        )

    R = T[:n_cutoff, :n_cutoff]
    N = T[n_cutoff:, n_cutoff:]
    C = T[:n_cutoff, n_cutoff:]

    W = np.eye(n, dtype=complex)

    if np.linalg.norm(C, np.inf) > tol:
        print(
            "Drazin inverse computation: Non-zero coupling block detected. Using Sylvester."
        )

        W_nz = solve_sylvester(R, -N, -C)
        W[:n_cutoff, n_cutoff:] = W_nz


def drazin_schur_2(A, tol=0, ax_plot=None, x_value=None):
    """Compute the Drazin inverse of a matrix A.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix.
    tol : float, optional
        Tolerance for determining the rank (default is 0).
    x_value: float, optional
        x value(s) to plot the eigenvalues at, if multiple cases are to be compared in one plot.
        If None (default), the eigenvalues are plotted at their index
    Returns
    -------
    np.ndarray
        The Drazin inverse of the matrix A.
    """
    n = A.shape[0]

    T, Z, n_cutoff = schur(A, output="complex", sort=lambda x: abs(x) > tol)

    if ax_plot is not None:
        eigenvalues = np.diag(T)

        if x_value is None:
            x_vals = np.arange(len(eigenvalues))
        else:
            x_vals = x_value * np.ones_like(eigenvalues)
        # eigenvalues_plot = np.zeros_like(eigenvalues)
        # for k, eigenvalue in enumerate(eigenvalues):
        #     eigenvalues_plot[k] = round_to_significant_digits(eigenvalue, 2)

        # _, idx_unique = np.unique(eigenvalues_plot, return_index=True)
        ax_plot.semilogy(
            x_vals,
            np.abs(eigenvalues),  # [idx_unique]),
            "x",
        )

    R = T[:n_cutoff, :n_cutoff]
    N = T[n_cutoff:, n_cutoff:]
    C = T[:n_cutoff, n_cutoff:]

    W = np.eye(n, dtype=complex)

    if np.linalg.norm(C, np.inf) > tol:
        print(
            "Drazin inverse computation: Non-zero coupling block detected. Using Sylvester."
        )

        W_nz = solve_sylvester(R, -N, -C)
        W[:n_cutoff, n_cutoff:] = W_nz

    # if np.max(np.abs(np.linalg.eig(N)[0])) > tol:
    #     warnings.warn(
    #         "Drazin inverse computation: Non-nilpotent block detected. Results may be inaccurate."
    #     )

    # if np.linalg.norm(Z @ Z.T.conj() - np.eye(n), np.inf) > tol:
    #     warnings.warn(
    #         "Drazin inverse computation: Schur vectors are not unitary. Results may be inaccurate."
    #     )

    drazin_schur = np.zeros_like(T)
    drazin_schur[:n_cutoff, :n_cutoff] = solve_triangular(
        R, np.eye(n_cutoff), lower=False
    )

    return Z @ W @ drazin_schur @ solve_triangular(W, Z.T.conj()), n - n_cutoff


def generalized_exponential(
    M, hill_matrix, t, tol_drazin=1e-6, tol_cond=1e6, a_pencil=None
):
    """Compute the generalized matrix exponential for DAEs based on the Drazin inverse.
    This yields the fundamental solution matrix for the LTI DAE

    M*z_dot = hill_matrix * z

    Parameters
    ----------
    hill_matrix : np.ndarray
        The Hill matrix of the DAE system.
    t : float
        The time at which to evaluate the exponential.
    """

    a_vals = [1.0, 10.0, 0.1, 100, 0.01, 1000, 0.001]
    if a_pencil is not None:
        a_vals = [a_pencil] + a_vals
    success = False
    for a in a_vals:
        pencil = a * M - hill_matrix
        if np.linalg.cond(pencil) < tol_cond:
            success = True
            break
    if not success:
        raise RuntimeError(
            f"Could not find suitable scaling factor 'a' for Drazin inverse with condition < {tol_cond}."
        )

    pencil_lu = lu_factor(a * M - hill_matrix)
    pencil_H = lu_solve(pencil_lu, hill_matrix)
    pencil_M = lu_solve(pencil_lu, M)
    pencil_drazin, ratio = drazin(pencil_M, tol_drazin)

    P_0 = pencil_drazin @ pencil_M
    exp = expm((pencil_drazin @ pencil_H) * t)

    return exp @ P_0, P_0, a
