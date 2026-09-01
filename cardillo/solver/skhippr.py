import numpy as np

from cardillo.math.approx_fprime import approx_fprime
from cardillo.solver.solution import Solution

# --- Solver ---
from skhippr.equations.EquationSystem import EquationSystem
from skhippr.solvers.newton import NewtonSolver
from skhippr_tmp.AbstractODE import AbstractDAE

# --- Continuation ---
from skhippr.solvers.continuation import pseudo_arclength_continuator
from skhippr.Fourier import Fourier
from skhippr_tmp.hbm import HBMEquationDAE


class CardilloSkhipprInterface(AbstractDAE):
    def __init__(
        self, cardillo_system, set_parameter_function=lambda param: None, param=None
    ):
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
            M_is_constant=not np.any(cardillo_system.I_M),
            invertible=True,
        )

        assert cardillo_system.nla_gamma == 0
        assert cardillo_system.nla_c == 0
        assert cardillo_system.nla_tau == 0
        assert cardillo_system.ntau == 0
        assert cardillo_system.nla_N == 0
        assert cardillo_system.nla_F == 0

        self.t = cardillo_system.t0
        la_S0 = np.zeros(cardillo_system.nla_S)
        self.x = np.concatenate(
            [cardillo_system.q0, cardillo_system.u0, cardillo_system.la_g0, la_S0]
        )
        self.cardillo_system = cardillo_system
        self._nq = cardillo_system.nq
        self._nu = cardillo_system.nu

        self.set_parameter = set_parameter_function
        if param is not None:
            self._param = param
            self.param = param

    @property
    def param(self):
        return self._param

    @param.setter
    def param(self, value):
        self._param = value
        self.set_parameter(np.squeeze(value))

    def M_small(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        t = float(np.squeeze(t))
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
        t = float(np.squeeze(t))
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

    def df_dparam(self, t=None, x=None):
        param0 = self.param

        def f_param(param):
            self.param = param
            f = self.dynamics(t, x)
            return f

        df_dparam = approx_fprime(param0, f_param, method="2-point")
        self.param = param0
        return df_dparam.reshape(-1, 1)

    def df_dx(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        t = float(np.squeeze(t))

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
            case "param":
                return self.df_dparam(t, x)

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

    def unpack(self, t=None, x=None):
        if x is None:
            x = self.x
        if t is None:
            t = self.t
        t = float(np.squeeze(t))
        self.check_dimensions(t, x)

        q, u, la_g, la_S = np.array_split(x, self.split_x)
        u_dot = None
        la_gamma = None
        la_c = None
        la_N = None
        la_F = None

        return dict(
            t=t,
            q=q,
            u=u,
            u_dot=u_dot,
            la_g=la_g,
            la_gamma=la_gamma,
            la_c=la_c,
            la_N=la_N,
            la_F=la_F,
        )


class SkhipprStaticContinuation:
    def __init__(
        self,
        cardillo_system,
        t1=1.0,
        skhippr_solver: NewtonSolver = None,
        min_step_size=0.001,
        max_step_size=0.01,
        verbose=True,
    ):
        self.system = cardillo_system
        self.t0 = cardillo_system.t0
        self.t1 = t1
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.verbose = verbose

        self.interface = CardilloSkhipprInterface(cardillo_system)
        self.interface.t = self.t0

        self.equation_sys = EquationSystem(
            equations=[self.interface],
            unknowns=["x"],
            equation_determining_stability=None,
        )

        if skhippr_solver is None:
            skhippr_solver = NewtonSolver(verbose=False)
        self.skhippr_solver = skhippr_solver

    def solve(self):
        if self.t1 == self.t0:
            self.skhippr_solver.solve(self.equation_sys)
            assert self.equation_sys.solved
            branch = [self.equation_sys]
            results = [self.interface.unpack()]

        else:
            results = []
            branch = []

            # --- Iterate through the branch ---
            for branch_point in pseudo_arclength_continuator(
                initial_system=self.equation_sys,
                solver=self.skhippr_solver,
                stepsize=self.max_step_size,
                stepsize_range=(self.min_step_size, self.max_step_size),
                continuation_parameter="t",
                initial_direction=(self.t1 - self.t0),
                verbose=self.verbose,
                num_steps=np.inf,
            ):
                branch.append(branch_point)
                results.append(self.interface.unpack(branch_point.t, branch_point.x))
                # break if t exceeds maximum
                if branch_point.t >= self.t1:
                    break

        # create cardillo solution
        results = {key: np.array([r[key] for r in results]) for key in results[0]}
        results["t_export"] = np.arange(len(results["t"]))
        return Solution(self.system, **results), branch


class SkhipprHBM:
    def __init__(
        self,
        cardillo_system,
        omega,
        N_HBM=5,
        L_DFT=512,
        initial_guess_time=None,
        skhippr_solver: NewtonSolver = None,
        newton_max_iter=20,
        verbose=True,
    ):
        self.system = cardillo_system
        self.L_DFT = L_DFT

        self.verbose = verbose

        self.interface = CardilloSkhipprInterface(cardillo_system)

        fourier = Fourier(
            N_HBM,
            L_DFT,
            self.interface.n_dof,
        )

        if initial_guess_time is None:
            initial_guess = np.zeros(self.interface.n_dof * (2 * N_HBM + 1))
            initial_guess[: self.interface.n_dof] = self.interface.x
        else:
            # TODO: initial_guess as cardillo solution for one period
            # initial_guess_time: np.ndarray(self.interface.n_dof, L_DFT)
            initial_guess = fourier.DFT(initial_guess_time)

        self.hbm = HBMEquationDAE(
            self.interface,
            omega,
            fourier,
            initial_guess=initial_guess,
            stability_method=None,  # TODO: KoopmanHillDAE
        )

        if skhippr_solver is None:
            skhippr_solver = NewtonSolver(
                max_iterations=newton_max_iter, verbose=verbose
            )
        self.skhippr_solver = skhippr_solver

    def solve(self):
        self.skhippr_solver.solve_equation(self.hbm, "X")

        t = self.hbm.fourier.time_samples(self.hbm.omega)
        x = self.hbm.x_time()

        results = [self.interface.unpack(ti, xi) for ti, xi in zip(t, x.T)]

        # create cardillo solution
        results = {key: np.array([r[key] for r in results]) for key in results[0]}
        return Solution(self.system, **results), self.hbm


class SkhipprHBMContinuation:
    def __init__(
        self,
        cardillo_system,
        omega,
        set_parameter_function,
        start_param,
        end_param,
        N_HBM=5,
        L_DFT=512,
        initial_guess_time=None,
        min_step_size=0.001,
        max_step_size=0.01,
        skhippr_solver: NewtonSolver = None,
        newton_max_iter=20,
        verbose=True,
        parameter_is_omega=False,
    ):
        assert not parameter_is_omega, "TODO"
        self.system = cardillo_system
        self.L_DFT = L_DFT

        self.verbose = verbose

        self.interface = CardilloSkhipprInterface(
            cardillo_system, set_parameter_function, start_param
        )
        self.start_param = start_param
        self.end_param = end_param

        self.min_step_size = min_step_size
        self.max_step_size = max_step_size

        fourier = Fourier(
            N_HBM,
            L_DFT,
            self.interface.n_dof,
        )

        if initial_guess_time is None:
            initial_guess = np.zeros(self.interface.n_dof * (2 * N_HBM + 1))
            initial_guess[: self.interface.n_dof] = self.interface.x
        else:
            # TODO: initial_guess as cardillo solution for one period
            # initial_guess_time: np.ndarray(self.interface.n_dof, L_DFT)
            initial_guess = fourier.DFT(initial_guess_time)

        hbm = HBMEquationDAE(
            self.interface,
            omega,
            fourier,
            initial_guess=initial_guess,
            stability_method=None,  # TODO: KoopmanHillDAE
        )

        self.hbm = EquationSystem(
            [hbm],
            ["X"],
            hbm,
        )

        if skhippr_solver is None:
            skhippr_solver = NewtonSolver(max_iterations=newton_max_iter, verbose=False)
        self.skhippr_solver = skhippr_solver

    def solve(self):
        solutions = []
        results = []
        branch = []

        # --- Iterate through the branch ---
        for branch_point in pseudo_arclength_continuator(
            initial_system=self.hbm,
            solver=self.skhippr_solver,
            stepsize=self.max_step_size,
            stepsize_range=(self.min_step_size, self.max_step_size),
            continuation_parameter="param",
            initial_direction=(self.end_param - self.start_param),
            verbose=self.verbose,
            num_steps=np.inf,
        ):
            branch.append(branch_point)

            t = branch_point.equations[0].fourier.time_samples(
                branch_point.equations[0].omega
            )
            x = branch_point.equations[0].x_time()
            results = [self.interface.unpack(ti, xi) for ti, xi in zip(t, x.T)]
            results = {key: np.array([r[key] for r in results]) for key in results[0]}
            solutions.append(
                Solution(self.system, **results, param=np.squeeze(branch_point.param))
            )

            # break if t exceeds maximum
            if branch_point.param >= self.end_param:
                break

        return solutions, branch

        self.skhippr_solver.solve_equation(self.hbm, "X")

        t = self.hbm.fourier.time_samples(self.hbm.omega)
        x = self.hbm.x_time()

        results = [self.interface.unpack(ti, xi) for ti, xi in zip(t, x.T)]

        # create cardillo solution
        results = {key: np.array([r[key] for r in results]) for key in results[0]}
        return Solution(self.system, **results), self.hbm
