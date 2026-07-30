"""Algebraic equation system. Encodes a system of algebraic equations. Base class."""

from abc import ABC, abstractmethod

from typing import override
from collections.abc import Callable
import numpy as np
from copy import copy
import warnings

from skhippr.stability.AbstractStabilityMethod import StabilityEquilibrium
from skhippr.equations.AbstractEquation import AbstractEquation

from skhippr.odes.AbstractODE import AbstractODE


class AbstractDAE(AbstractODE):
    """Abstract base class for differential-algebraic equations (DAEs). DAEs consist of differential equations coupled with algebraic constraint equations. The DAE is formulated in the form

        M*x_dot = f(t, x)

    with a non-invertible matrix M.
    The equilibrium problem can immediately solved by passing the DAE into the :py:class:`~skhippr.solvers.newton.NewtonSolver`.

    Attributes:
    -----------

    autonomous : bool
        Whether the DAE is autonomous (does not depend on time).
    n_dof : int
        Number of degrees of freedom of the DAE.
    n_constraints : int
        Number of algebraic constraints in the DAE.
    x : np.ndarray
        State vector.
    t : float
        Time variable.
    """

    def __init__(
        self,
        n_dof,
        autonomous: bool,
        stability_method=None,
        M_is_constant=False,
        invertible: bool = False,
    ):

        super().__init__(autonomous, n_dof, stability_method)
        self.M_is_constant = M_is_constant
        self.invertible = invertible

    @abstractmethod
    def M_small(self, t=None, x=None):
        if t is None:
            t = self.t
        if x is None:
            x = self.x

        return np.eye(self.n_dof)
