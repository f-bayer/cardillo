"""
Find and plot the equilibrium of the Truss system using SKHiPPR.

"""

import numpy as np
import matplotlib.pyplot as plt

# --- Solver ---
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.newton import NewtonSolver
from skhippr.odes.AbstractODE import AbstractODE
from skhippr_tmp.AbstractODE import AbstractDAE

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator, BranchPoint

# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.equilibria import plot_equilibrium, plot_eigenvalues
import numpy as np

from cardillo import System


class TrussSubSystem:
    def __init__(self, k, l0, a, F, c, m, q0, u0):
        self.k = k
        self.l0 = l0
        self.a = a
        self.F = F
        self.c = c

        self.nq = 1
        self.nu = 1

        self.q0 = q0
        self.u0 = u0
        self._M = np.array([[m]])

    def q_dot(self, t, q, u):
        return u

    def q_dot_u(self, t, q):
        return np.array([[1.0]])

    def M(self, t, q):
        return self._M

    def h(self, t, q, u):
        return (
            -self.k * q
            + self.k * q * self.l0 / np.sqrt(self.a**2 + q**2)
            + self.F
            - self.c * u
        )

    def h_q(self, t, q, u):
        return (
            -self.k
            + self.k * self.l0 / np.sqrt(self.a**2 + q**2)
            - self.k * self.l0 * q**2 / (np.sqrt(self.a**2 + q**2) ** 3)
        )

    def h_u(self, t, q, u):
        return -self.c

    def set_parameter(self, p):
        self.F = p


def F_fun(t):
    return -0.5 + t


class TrussCardilloSkhipprInterface(AbstractDAE):
    def __init__(self, cardillo_system, param):
        super().__init__(
            autonomous=True,
            n_dof=cardillo_system.nq + cardillo_system.nu,
            stability_method=None,
            M_is_constant=True,
            invertible=True,
        )
        self.t = cardillo_system.t0
        self.x = np.concatenate([cardillo_system.q0, cardillo_system.u0])
        self.cardillo_system = cardillo_system
        self._nq = cardillo_system.nq
        self._nu = cardillo_system.nu
        self._param = param
        self.param = param

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value
        self.cardillo_system.set_parameter(value)

    def M_small(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)
        q = x[: self._nq]
        m1 = np.eye(self._nq)
        M = self.cardillo_system.M(t, q).toarray()
        return np.block([[m1, None], [None, M]])

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q, u = x[: self._nq], x[self._nq :]
        q_dot = self.cardillo_system.q_dot(t, q, u)
        h = self.cardillo_system.h(t, q, u)
        return np.concatenate([q_dot, h])

    def df_dx(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t

        q, u = x[: self._nq], x[self._nq :]

        f_x = np.zeros((self.n_dof, self.n_dof))
        f_x[: self._nq, : self._nq] = self.cardillo_system.q_dot_q(t, q, u).toarray()
        f_x[: self._nq, self._nq :] = self.cardillo_system.q_dot_u(t, q).toarray()
        f_x[self._nq :, : self._nq] = self.cardillo_system.h_q(t, q, u).toarray()
        f_x[self._nq :, self._nq :] = self.cardillo_system.h_u(t, q, u).toarray()
        return f_x

    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                return self.df_dx(t, x)
            # case "t":
            #     return np.array([[0.0], [1.0]])

            # case "F":
            #     return self.df_dF(x)
            # case "k":
            #     return self.df_dk(x)
            # case "c":
            #     return self.df_dc(x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )


def main():
    truss_subsystem = TrussSubSystem(
        k=3.0,
        l0=1.2,
        a=1.0,
        F=0.0,
        c=0.5,
        m=1.0,
        q0=np.array([-1.0]),
        u0=np.array([0.0]),
    )

    cardillo_system = System()
    cardillo_system.add(truss_subsystem)
    cardillo_system.assemble()

    # truss_interface = TrussCardilloSkhipprInterface(cardillo_system)

    """
    Demonstrates solving an equation for an equilibrium and visualizes the equilibrium as well as its eigenvalues using SKHiPPR visualization methods.
    Performs continuation over the parameter `F` and then visualizes the branch with its stability intervals.

    This function instantiates a :py:class:`~skhippr.odes.autonomous.Truss` object as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE` and solves it with either :py:func:`skhippr.solvers.newton.NewtonSolver.solve_equation` or with :py:func:`skhippr.solvers.newton.NewtonSolver.solve` after embedding it into an :py:class:`~skhippr.equations.EquationSystem.EquationSystem`.

    It generates 3 figures:

    #. A plot of the ode equilibrium
    #. A plot of the equilibrium eigenvalues visualizing its stability
    #. A continuation plot highlighting the fold bifurcations inherent to the system.

    """
    # --- Instantiation of the ODE at initial point ---
    # ode = Truss(x=[1.0, 2.0], F=-0.5, a=1.0, l_0=1.2, k=3.0, m=1.0, c=0.5)
    ode = TrussCardilloSkhipprInterface(cardillo_system, -0.5)
    solver = NewtonSolver(verbose=True)

    # --- ODEs can be packed into an EquationSystem for solving ---
    equation_sys = EquationSystem(
        equations=[ode], unknowns=["x"], equation_determining_stability=ode
    )
    solver.solve(equation_sys)

    # --- or passed to a solver directly using a different method ---
    solver.solve_equation(equation=ode, unknown="x")

    # --- Standard SKHiPPR visualization calls create and return new figures for each plot ---
    plot_equilibrium(ode=ode)

    # --- SKHiPPR visualization methods accept EquationSystem objects that contain an AbstractODE as well ---
    plot_eigenvalues(ode=equation_sys)

    branch: list[BranchPoint] = []

    # --- Iterate through the branch ---
    for branch_point in pseudo_arclength_continuator(
        initial_system=equation_sys,
        solver=solver,
        stepsize=0.01,
        stepsize_range=(0.001, 0.01),
        continuation_parameter="param",
        initial_direction=1,
        verbose=False,
        num_steps=400,
    ):
        branch.append(branch_point)
        # break if F exceeds maximum
        if branch_point.param > 0.5:
            break

    # --- Plot the continuation curve ---
    plot_continuation(branch, marker="x")


if __name__ == "__main__":
    main()
    plt.show()
