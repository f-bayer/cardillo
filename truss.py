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
from skhippr.visualization.cycles import plot_period, plot_phase
from skhippr.Fourier import Fourier
from skhippr_tmp.hbm import HBMEquationDAE


from cardillo import System
from cardillo.discrete import RigidBody, PointMass
from cardillo.constraints import Prismatic
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.force_laws import Spring, KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force


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
        self.split_x = np.cumsum(
            np.array(
                [
                    cardillo_system.nq,
                    cardillo_system.nu,
                    cardillo_system.nla_g,
                    cardillo_system.nla_S,
                ],
                dtype=int,
            )
        )

        n_dof = self.split_x[-1]
        self.split_x = self.split_x[:-1]

        super().__init__(
            autonomous=True,
            n_dof=n_dof,
            stability_method=None,
            M_is_constant=True,
            invertible=True,
        )
        self.t = cardillo_system.t0
        la_S0 = np.zeros(cardillo_system.nla_S)
        self.x = np.concatenate(
            [cardillo_system.q0, cardillo_system.u0, cardillo_system.la_g0, la_S0]
        )
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
        M_ret = np.zeros((self.n_dof, self.n_dof))

        q = x[: self._nq]

        M = self.cardillo_system.M(t, q).toarray()

        M_ret[: self.split_x[0], : self.split_x[0]] = np.eye(self._nq)

        M_ret[self.split_x[0] : self.split_x[1], self.split_x[0] : self.split_x[1]] = M

        return M_ret

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        self.check_dimensions(t, x)

        q, u, la_g, la_S = np.array_split(x, self.split_x)

        q_dot = (
            self.cardillo_system.q_dot(t, q, u)
            + self.cardillo_system.g_S_q(t, q, format="csc").T @ la_S
        )
        W_g = self.cardillo_system.W_g(t, q, format="csr")

        h = self.cardillo_system.h(t, q, u) + W_g @ la_g
        g = self.cardillo_system.g(t, q)
        g_S = self.cardillo_system.g_S(t, q)
        return np.concatenate([q_dot, h, g, g_S])

    def df_dx(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t

        q, u, la_g, la_S = np.array_split(x, self.split_x)
        W_g = self.cardillo_system.W_g(t, q).toarray()

        g_S_q = self.cardillo_system.g_S_q(t, q)
        f_x = np.zeros((self.n_dof, self.n_dof))
        split_x = self.split_x
        # TODO: the derivative of g_S_q.T @ la_S w.r.t q is missing below
        f_x[: split_x[0], : split_x[0]] = self.cardillo_system.q_dot_q(
            t, q, u
        ).toarray()
        f_x[: split_x[0], split_x[0] : split_x[1]] = self.cardillo_system.q_dot_u(
            t, q
        ).toarray()
        f_x[: split_x[0], split_x[2] :] = g_S_q.T.toarray()
        f_x[split_x[0] : split_x[1], : split_x[0]] = (
            self.cardillo_system.h_q(t, q, u).toarray()
            + self.cardillo_system.Wla_g_q(t, q, la_g).toarray()
        )
        f_x[split_x[0] : split_x[1], split_x[0] : split_x[1]] = (
            self.cardillo_system.h_u(t, q, u).toarray()
        )
        f_x[split_x[0] : split_x[1], split_x[1] : split_x[2]] = W_g
        f_x[split_x[1] : split_x[2], : split_x[0]] = self.cardillo_system.g_q(
            t, q
        ).toarray()
        f_x[split_x[2] :, : split_x[0]] = g_S_q.toarray()
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
    a = 1.0
    mass = 1.0
    l0 = 1.2
    k_spring = 3.0
    d_damper = 0.5

    cardillo_system = System()

    # rigid body
    # rb = PointMass(mass=mass)
    # con = ProjectedPositionOrientationBase(cardillo_system.origin, rb, [1, 2], [])

    rb = RigidBody(mass, B_Theta_C=np.eye(3))
    con = Prismatic(cardillo_system.origin, rb, axis=0)

    # spring
    inter_spring = TwoPointInteraction(
        cardillo_system.origin, rb, B_r_CP1=np.array([0, 0, a])
    )
    spring = Spring(inter_spring, k=k_spring, l_ref=l0, compliance_form=False)

    # damper
    inter_damper = TwoPointInteraction(
        cardillo_system.origin, rb, B_r_CP1=np.array([a, 0, 0])
    )
    damper = KelvinVoigtElement(inter_damper, k=0.0, d=d_damper, compliance_form=False)

    # force
    force = Force(np.zeros(3), rb)

    cardillo_system.add(rb, con, spring, damper, force)

    # handle force parameter update for skhippr
    def set_force_parameter(F):
        force.force = lambda t, F=F: np.array([F * np.sin(2 * np.pi * t) - 0.5, 0, 0])

    force.set_parameter = set_force_parameter

    cardillo_system.assemble()

    # truss_interface = TrussCardilloSkhipprInterface(cardillo_system)

    # --- Instantiation of the DAE at initial point ---
    # ode = Truss(x=[1.0, 2.0], F=-0.5, a=1.0, l_0=1.2, k=3.0, m=1.0, c=0.5)
    cardillo_interface = TrussCardilloSkhipprInterface(cardillo_system, -0.5)
    cardillo_interface.param = 0.0

    solver = NewtonSolver(verbose=True)

    # --- solve static equilibrium as initial condition ---
    print("Solving static equilibrium as initial condition...")
    solver.solve_equation(equation=cardillo_interface, unknown="x")

    # --- Plot static equilibrium ---
    plot_equilibrium(
        ode=cardillo_interface,
        idx=[0, cardillo_interface.n_dof - 5],
        title="initial equilibrium",
    )

    # --- Try HBM ---
    n_hbm = 2
    l_dft = 512
    fourier = Fourier(
        n_hbm,
        l_dft,
        cardillo_interface.n_dof,
    )
    initial_guess = np.zeros(
        cardillo_interface.n_dof * (2 * n_hbm + 1)
    ) + 1e-3 * np.random.rand(cardillo_interface.n_dof * (2 * n_hbm + 1))
    # initial_guess[0] = -1
    initial_guess[: cardillo_interface.n_dof] = cardillo_interface.x

    hbm_dae = HBMEquationDAE(
        cardillo_interface,
        2 * np.pi,
        fourier,
        initial_guess=initial_guess,
        stability_method=None,
    )

    # --- ODEs can be packed into an EquationSystem for solving ---
    equation_sys = EquationSystem(
        equations=[hbm_dae],
        unknowns=["X"],
        equation_determining_stability=hbm_dae,
    )
    solver.solve(equation_sys)

    plot_period(hbm_dae, title=f"initial periodic solution (param = {hbm_dae.param})")
    plot_phase(
        hbm_dae,
        idx=[0, cardillo_interface._nq],
        title=f"initial periodic solution (param = {hbm_dae.param})",
    )

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

    def plot_fun(bp):
        return (bp.equations[0].param, bp.equations[0].X[cardillo_interface.n_dof])

    # --- Plot the continuation curve ---
    plot_continuation(
        branch,
        marker="x",
        plot_fun=plot_fun,
        xlabel="parameter (F)",
        ylabel="1st cosine amplitude of q",
    )


if __name__ == "__main__":
    main()
    plt.show()
