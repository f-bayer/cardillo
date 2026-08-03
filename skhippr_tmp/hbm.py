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

    def error_bound_fundamental_matrix(self, t=None, _as=None, bs=None):
        raise NotImplementedError(
            "Error bounds for the fundamental matrix not applicable to DAEs."
        )


class HBMSystem(EquationSystem):
    """This subclass of :py:class:`~skhippr.equations.EquationSystem.EquationSystem` instantiates a :py:class:`~skhippr.cycles.hbm.HBMEquation` and considers it as the first equation. The Fourier coefficient vector ``X`` is the first unknown.

    If the underlying ODE is autonomous, the frequency ``omega`` of the periodic solution is not known in advance and is appended to the unknowns. Correspondingly, a :py:class:`~skhippr.cycles.hbm.HBMPhaseAnchor` equation is appended to the equations.
    """

    def __init__(
        self,
        ode,
        omega,
        fourier,
        initial_guess: np.ndarray = None,
        period_k: float = 1,
        stability_method: "AbstractStabilityHBM" = None,
        harmo_anchor: int = 1,
        dof_anchor: int = 0,
    ):
        hbm = HBMEquation(
            ode,
            omega,
            fourier,
            initial_guess,
            period_k,
            stability_method=stability_method,
        )

        equations = [hbm]
        unknowns = ["X"]

        if ode.autonomous:
            unknowns.append("omega")
            anchor_equation = HBMPhaseAnchor(
                fourier=hbm.fourier, X=hbm.X, harmo=harmo_anchor, dof=dof_anchor
            )
            equations.append(anchor_equation)

        super().__init__(
            equations=equations, unknowns=unknowns, equation_determining_stability=hbm
        )


class HBMPhaseAnchor(AbstractEquation):
    """This class implements an anchor equation for the harmonic balance method (HBM) in autonomous systems to ensure that the phase of a specified degree of freedom and harmonic of the periodic solution does not change during the HBM solution procedure.

    * Complex formulation:
        exp(i*phi) = X+/X- = const
    * Real formulation:
        -tan(phi) = c_k/s_k = const

    Hereby, ``harmo`` and ``dof``   specify the harmonic and degree of freedom for which the phase is anchored.

    """

    def __init__(self, fourier, X, harmo, dof):
        super().__init__(None)
        self.X = X
        self.idx_anchor = self._determine_anchor(fourier, harmo, dof)
        self.anchor = np.zeros((1, X.size), dtype=X.dtype)
        self.anchor[0, self.idx_anchor[0]] = -1
        # self.phase_required = self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]

    def residual_function(self):
        """Always returns zero."""
        # anchor equation (phase may not change):
        # delta X[anchor[0]] = X[anchor[0]]/X[anchor[1]] * delta X[anchor[1]]

        # phase = self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]
        # return phase - self.phase_required
        return np.atleast_1d(0)

    def closed_form_derivative(self, variable):
        """Return the anchor as derivative w.r.t ``X`` and zero otherwise."""
        if variable == "X":
            self.anchor[0, self.idx_anchor[1]] = (
                self.X[self.idx_anchor[0]] / self.X[self.idx_anchor[1]]
            )
            return self.anchor
        else:
            return np.atleast_2d(0)

    def _determine_anchor(self, fourier, harmo: int = 1, dof: int = 0) -> np.ndarray:
        """Determine the index of the anchor equation.
        The anchor equation ensures that the phase of the  harmo-th harmonic
        and the dof-th degree of freedom does not change during HBM solution for autonomous systems.
        """
        if fourier.real_formulation:
            # -tan(phi) = c_k/s_k = const -->  delta c = c_k/s_k * delta s
            idx_anchor = [
                harmo * fourier.n_dof + dof,
                (harmo + fourier.N_HBM) * fourier.n_dof + dof,
            ]
        else:
            # exp(i*phi) = X+/X- = const -->  delta X+ = X+/X- * delta X-
            idx_anchor = [
                (fourier.N_HBM + harmo) * fourier.n_dof + dof,
                (fourier.N_HBM - harmo) * fourier.n_dof + dof,
            ]

        # Avoid division by zero
        if abs(self.X[idx_anchor[1]]) < (1e-4 * abs(self.X[idx_anchor[0]])):
            idx_anchor.reverse()

        return np.array(idx_anchor)
