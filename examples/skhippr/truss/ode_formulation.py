import numpy as np

from skhippr.equations.EquationSystem import EquationSystem
from skhippr.odes.AbstractODE import AbstractODE
from cardillo.solver.skhippr_tmp.AbstractODE import AbstractDAE


class Truss(AbstractODE):
    """
    Autonomous truss system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A mass ``m`` can move horizontally with viscous damping (``c``). It is attached to a linear spring (``k``, ``l_0``) mounted at the point ``(0, a)``. A constant force ``F`` acts on the mass. For small values of ``F``, there are three coexisting equilibria. The equations of motion are ::

        dx[0]/dt = x[1]
        dx[1]/dt = -k/m * x[0] + k/m * x[0] * l_0 / sqrt(a**2 + x[0]**2) + F/m - c/m * x[1]

    """

    def __init__(
        self,
        x: np.ndarray,
        k: float,
        c: float,
        F: float,
        a: float,
        l_0: float,
        m: float,
        omega: float,
    ):
        super().__init__(autonomous=False, n_dof=2)
        self.x = x
        self.k = k
        self.c = c
        self.F = F
        self.a = a
        self.l_0 = l_0
        self.m = m
        self.omega = omega

    def F_fun(self, t):
        return self.F * np.sin(self.omega * t) - 0.5

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        if t is None:
            t = self.t

        self.check_dimensions(t=t, x=x)

        q = x[0, ...]
        q_dot = x[1, ...]

        f = np.zeros_like(x)
        f[0, ...] = q_dot
        f[1, ...] = -self.k / self.m * q
        f[1, ...] += self.k / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        f[1, ...] = f[1, ...] + self.F_fun(t) / self.m - self.c / self.m * q_dot

        return f

    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x

        if t is None:
            t = self.t

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                return self.df_dx(x)
            case "F":
                return self.df_dF(t, x)
            case "k":
                return self.df_dk(x)
            case "c":
                return self.df_dc(x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dF(self, t, x=None):
        if x is None:
            x = self.x
        df_dF = np.zeros_like(x)
        df_dF[1, ...] = 1 / self.m * np.sin(self.omega * t)
        return df_dF[:, np.newaxis, ...]

    def df_dc(self, x=None):
        if x is None:
            x = self.x
        df_dc = np.zeros_like(x)
        df_dc[1, ...] = -x[1, ...] / self.m
        return df_dc[:, np.newaxis, ...]

    def df_dk(self, x=None):
        if x is None:
            x = self.x
        q = x[0, ...]
        df_dk = np.zeros_like(x)
        df_dk[1, ...] = -1 / self.m * q
        df_dk[1, ...] += 1 / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        return df_dk[:, np.newaxis, ...]

    def df_dx(self, x=None):
        if x is None:
            x = self.x

        q = x[0, ...]

        df_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]))
        df_dx[0, 1, ...] = 1
        df_dx[1, 1, ...] = -self.c / self.m
        df_dx[1, 0, ...] = -self.k / self.m
        df_dx[1, 0, ...] += self.k / self.m * self.l_0 / np.sqrt(self.a**2 + q**2)
        df_dx[1, 0, ...] -= (
            self.k / self.m * self.l_0 * q**2 / (np.sqrt(self.a**2 + q**2) ** 3)
        )
        return df_dx


class Truss_compliance(AbstractDAE):
    """
    Autonomous truss system as a subclass of :py:class:`~skhippr.odes.AbstractODE.AbstractODE`. A mass ``m`` can move horizontally with viscous damping (``c``). It is attached to a linear spring (``k``, ``l_0``) mounted at the point ``(0, a)``. A constant force ``F`` acts on the mass. For small values of ``F``, there are three coexisting equilibria. The equations of motion are ::

        dx[0]/dt = x[1]
        dx[1]/dt = -k/m * x[0] + k/m * x[0] * l_0 / sqrt(a**2 + x[0]**2) + F/m - c/m * x[1]

    """

    def __init__(
        self,
        x: np.ndarray,
        k: float,
        c: float,
        F: float,
        a: float,
        l_0: float,
        m: float,
        omega: float,
    ):
        self._M_small = np.diag([1.0, 1.0, 0.0])
        x = np.concatenate([x, [-k * (np.sqrt(x[0] ** 2 + a**2) - l_0)]])

        super().__init__(
            autonomous=False, n_dof=3, M_is_constant=True, invertible=False
        )
        self.x = x
        self.k = k
        self.c = c
        self.F = F
        self.a = a
        self.l_0 = l_0
        self.m = m
        self.omega = omega

    def F_fun(self, t):
        return self.F * np.sin(self.omega * t) - 0.5

    def M_small(self, t=None, x=None):
        return self._M_small

    def dynamics(self, t=None, x=None):
        if x is None:
            x = self.x

        if t is None:
            t = self.t

        self.check_dimensions(t=t, x=x)

        q = x[0, ...]
        q_dot = x[1, ...]
        la_c = x[2, ...]
        l = np.sqrt(self.a**2 + q**2)

        f = np.zeros_like(x)
        f[0, ...] = q_dot
        f[1, ...] = la_c / self.m * q / l
        f[1, ...] = f[1, ...] + self.F_fun(t) / self.m - self.c / self.m * q_dot
        f[2, ...] = la_c / self.k + (l - self.l_0)

        return f

    def closed_form_derivative(self, variable, t=None, x=None):

        if x is None:
            x = self.x

        if t is None:
            t = self.t

        self.check_dimensions(t=t, x=x)

        match variable:
            case "x":
                return self.df_dx(x)
            case "F":
                return self.df_dF(t, x)
            # case "k":
            #     return self.df_dk(x)
            case "c":
                return self.df_dc(x)
            case _:
                raise NotImplementedError(
                    f"Derivative w.r.t {variable} not implemented in closed form."
                )

    def df_dF(self, t, x=None):
        if x is None:
            x = self.x
        df_dF = np.zeros_like(x)
        df_dF[1, ...] = 1 / self.m * np.sin(self.omega * t)
        return df_dF[:, np.newaxis, ...]

    def df_dc(self, x=None):
        if x is None:
            x = self.x
        df_dc = np.zeros_like(x)
        df_dc[1, ...] = -x[1, ...] / self.m
        return df_dc[:, np.newaxis, ...]

    def df_dk(self, x=None):
        raise NotImplementedError
        # update for compliance form
        if x is None:
            x = self.x
        q = x[0, ...]
        df_dk = np.zeros_like(x)
        df_dk[1, ...] = -1 / self.m * q
        df_dk[1, ...] += 1 / self.m * q * self.l_0 / np.sqrt(self.a**2 + q**2)
        return df_dk[:, np.newaxis, ...]

    def df_dx(self, x=None):
        if x is None:
            x = self.x

        q = x[0, ...]
        la_c = x[2, ...]
        l = np.sqrt(self.a**2 + q**2)
        l_q = q / l

        df_dx = np.zeros((x.shape[0], x.shape[0], *x.shape[1:]))
        df_dx[0, 1, ...] = 1
        df_dx[1, 0, ...] = la_c / self.m / l - la_c / self.m * q / l**2 * l_q
        df_dx[1, 1, ...] = -self.c / self.m
        df_dx[1, 2, ...] = 1.0 / self.m * q / l
        df_dx[2, 0, ...] = l_q
        df_dx[2, 2, ...] = 1 / self.k
        return df_dx


def run(compliance_form):
    omega = 0.01
    if compliance_form:
        truss = Truss_compliance(np.array([0.0, 0.0]), 3.0, 0.5, 0.0, 1.0, 1.2, 1.0, omega)
    else:
        truss = Truss(np.array([0.0, 0.0]), 3.0, 0.5, 0.0, 1.0, 1.2, 1.0, omega)

    fourier = Fourier(5, 512, truss.n_dof)

    initial_guess = np.zeros(fourier.n_dof * (2 * fourier.N_HBM + 1))
    initial_guess[: fourier.n_dof] = truss.x

    if compliance_form:
        _hbm = HBMEquationDAE(
            truss,
            omega,
            fourier,
            initial_guess=initial_guess,
            stability_method=KoopmanHillDAE(fourier, autonomous=False),
        )
        hbm = EquationSystem(
            [_hbm],
            ["X"],
            _hbm,
        )
    else:
        hbm = HBMSystem(
            truss,
            omega,
            fourier,
            initial_guess,
            stability_method=KoopmanHillSubharmonic(fourier),
        )

    # --- Iterate through the branch ---
    branch = []
    for branch_point in pseudo_arclength_continuator(
        initial_system=hbm,
        solver=NewtonSolver(),
        stepsize=0.01,
        stepsize_range=(0.001, 0.01),
        continuation_parameter="F",
        initial_direction=1,
        verbose=True,
        num_steps=1e3,
    ):
        branch.append(branch_point)
        # break if F exceeds maximum
        if branch_point.F > 1.0:
            break

    def plot_fun(bp):
        # maximum of q[0] over one period
        return (bp.equations[0].F, np.max(bp.equations[0].x_time()[0, :]))

    plot_continuation(
        branch,
        marker="x",
        plot_fun=plot_fun,
        xlabel="parameter (F)",
        ylabel="amplitude of q[0]",
    )

    return animate_phase(branch, idx=[0, 1], interval=60)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from skhippr.Fourier import Fourier
    from skhippr.cycles.hbm import HBMSystem
    from skhippr.stability.KoopmanHillProjection import KoopmanHillSubharmonic
    from skhippr.solvers.continuation import pseudo_arclength_continuator
    from skhippr.solvers.newton import NewtonSolver

    from skhippr.visualization.continuation import plot_continuation
    from skhippr.visualization.equilibria import plot_equilibrium, plot_eigenvalues
    from skhippr.visualization.cycles import plot_period, plot_phase, animate_phase

    from cardillo.solver.skhippr_tmp.stability import KoopmanHillDAE
    from cardillo.solver.skhippr_tmp.hbm import HBMEquationDAE

    _ = run(False)
    _ = run(True)

    plt.show()
