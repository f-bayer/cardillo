from typing import override, TYPE_CHECKING
import warnings
import numpy as np

from skhippr.equations.AbstractEquation import AbstractEquation
from skhippr.cycles.AbstractCycleEquation import AbstractCycleEquation
from skhippr.odes.AbstractODE import AbstractODE
from skhippr_tmp.AbstractODE import AbstractDAE
from skhippr.cycles.hbm import HBMEquation
from skhippr.Fourier import Fourier
from skhippr.equations.EquationSystem import EquationSystem

# Imports only needed for type hinting
if TYPE_CHECKING:
    from skhippr.stability.AbstractStabilityHBM import AbstractStabilityHBM


class HBMEquationDAE(HBMEquation):
    """This subclass of :py:class:`~skhippr.cycles.hbm.HBMEquation` is specifically designed to handle DAEs.

    It extends the differential part of the harmonic balance equations to account for the possibly non-invertible matrix M.
    Reference: Legrand2024 (https://hal.science/hal-04189699v2/file/JSD_Legrand_Pierre_2024.pdf)
    """

    def __init__(
        self,
        dae: AbstractDAE,
        omega: float,
        fourier: Fourier,
        initial_guess: np.ndarray = None,
        period_k: float = 1,
        stability_method=None,
    ):
        """
        Initialize the HBM equations for DAEs.
        """
        super().__init__(
            ode=dae,
            omega=omega,
            fourier=fourier,
            initial_guess=initial_guess,
            period_k=period_k,
            stability_method=stability_method,
        )

    def M(self):
        if self.ode.M_is_constant:
            return np.kron(np.eye(2 * self.fourier.N_HBM + 1), self.ode.M_small())
        else:
            M_samples = np.zeros((self.ode.n_dof, self.ode.n_dof, self.fourier.L_DFT))
            for k, (t, x) in enumerate(
                zip(self.fourier.time_samples(self.omega_solution), self.x_time().T)
            ):
                M_samples[:, :, k] = self.ode.M_small(t, x)
            return self.fourier.matrix_DFT(M_samples)

    def aft(self, X=None) -> np.ndarray:
        """
        Overwrite the HBM residual computation to account for the weight matrix M in DAEs.
        """

        R = super().aft(X)
        deriv = self.fourier.derivative_coeffs(X, self.omega_solution)
        # Remove the effect of direct differentiation and add the effect of M
        R += deriv - self.M() @ deriv

        return R

    def dR_domega(self, X=None):
        return self.M() @ super().dR_domega(X)

    def dR_dX(self, X=None):
        """
        Overwrite the HBM Jacobian to account for the weight matrix M in DAEs.
        """
        derivative = super().dR_dX(X)
        derivative += self.omega_solution * (
            self.fourier.derivative_matrix - self.M() @ self.fourier.derivative_matrix
        )
        return derivative

    @override
    def dR_dvar(self, variable, X=None) -> np.ndarray:
        """
        Overwrite the HBM during finite differences
        """

        try:
            return super().dR_dvar(variable, X)
        except (
            ValueError
        ):  # vectorization didn't work - use finite differences immediately in the freq domain
            return self.finite_difference_derivative(variable, h_step=1e-4)

    def error_bound_fundamental_matrix(self, t=None, _as=None, bs=None):
        raise NotImplementedError(
            "Error bounds for the fundamental matrix not applicable to DAEs."
        )
