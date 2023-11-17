import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import splu
from tqdm import tqdm

from cardillo.solver import SolverOptions, Solution, compute_I_F
from cardillo.math import prox_R0_np, prox_sphere, estimate_prox_parameter


class Moreau:
    def __init__(self, system, t1, dt, options=SolverOptions()):
        self.system = system
        self.options = options

        # integration time
        t0 = system.t0
        self.t1 = (
            t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        )
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.nq = self.system.nq
        self.nu = self.system.nu
        self.nla_g = self.system.nla_g
        self.nla_gamma = self.system.nla_gamma
        self.nla_N = self.system.nla_N
        self.nla_F = self.system.nla_F
        self.nR_smooth = self.nu + self.nla_g + self.nla_gamma
        self.nR = self.nR_smooth + self.nla_N + self.nla_F

        # connectivity matrix of normal force directions and friction force directions
        self.NF_connectivity = self.system.NF_connectivity

        # initial conditions
        self.tn = system.t0
        self.qn = system.q0
        self.un = system.u0
        self.q_dotn = system.q_dot0
        self.u_dotn = system.u_dot0
        self.la_c0 = system.la_c0
        la_g0 = system.la_g0
        la_gamma0 = system.la_gamma0
        la_N0 = system.la_N0
        la_F0 = system.la_F0

        # consistent initial percussion
        self.P_gn = la_g0 * dt
        self.P_gamman = la_gamma0 * dt
        self.P_Nn = la_N0 * dt
        self.P_Fn = la_F0 * dt

        self.split_x = np.array(
            [
                self.nu,
                self.nu + self.nla_g,
            ],
            dtype=int,
        )
        self.x = np.concatenate(
            (
                self.un,
                self.P_gn,
                self.P_gamman,
            )
        )
        self.x0 = self.x.copy()

    def prox(
        self, un1, P_N, P_F
    ):  # , lu_A, b, W_N, W_F, I_N, I_F, tn1, qn1, un, prox_r_N, prox_r_F, mu):
        # projection for contacts
        xi_N = self.W_N.T @ un1 + self.xi_N0
        P_N[self.I_N] = prox_R0_np(
            P_N[self.I_N] - self.prox_r_N[self.I_N] * xi_N[self.I_N]
        )

        # friction projection
        xi_F = self.W_F.T @ un1 + self.xi_F0
        mu = self.system.mu
        for i_N, i_F in enumerate(self.NF_connectivity):
            if self.I_N[i_N] and len(i_F):
                P_F[i_F] = prox_sphere(
                    P_F[i_F] - self.prox_r_F[i_N] * xi_F[i_F],
                    mu[i_N] * P_N[i_N],
                )

        return P_N, P_F

        # update rhs
        bb = b.copy()
        bb[: self.nu] += W_N[:, I_N] @ P_N[I_N] + W_F[:, I_F] @ P_F[I_F]

        # solve for new velocities and Lagrange multipliers of bilateral constraints
        self.x0 = self.x.copy()
        self.x = lu_A.solve(bb)

        return np.concatenate([P_N, P_F])

    def step(self):
        # general quantities
        dt = self.dt
        un = self.un
        tn1 = self.tn + dt
        self.tn12 = tn12 = self.tn + 0.5 * dt

        # explicit position update (midpoint) with projection
        self.qn12 = qn12 = self.qn + 0.5 * dt * self.system.q_dot(self.tn, self.qn, un)

        # get quantities from model
        M = self.system.M(tn12, qn12)
        h = self.system.h(tn12, qn12, un)
        W_g = self.system.W_g(tn12, qn12)
        W_gamma = self.system.W_gamma(tn12, qn12)
        W_c = self.system.W_c(tn12, qn12)
        la_c = self.system.la_c(tn12, qn12, un)
        chi_g = self.system.g_dot(tn12, qn12, np.zeros_like(un))
        chi_gamma = self.system.gamma(tn12, qn12, np.zeros_like(un))

        # Build matrix A for computation of new velocities and bilateral constraint percussions
        # M (uk1 - uk) - dt h - W_g P_g - W_gamma P_gamma - W_gN P_N - W_gT P_T = 0
        # -(W_g.T @ uk1 + chi_g) = 0
        # -(W_gamma.T @ uk1 + chi_gamma) = 0
        # fmt: off
        A = bmat([[         M, -W_g, -W_gamma], \
                  [    -W_g.T, None,     None], \
                  [-W_gamma.T, None,     None]], format="csc")
        # fmt: on

        # perform LU decomposition only once since matrix A is constant in
        # each time step saves alot work in the fixed point iteration
        lu_A = splu(A)

        # initial right hand side without contact forces
        b = np.concatenate(
            (
                M @ un + dt * (h + W_c @ la_c),
                chi_g,
                chi_gamma,
            )
        )

        # solve for initial velocities and percussions of the bilateral
        # constraints for the fixed point iteration
        x0 = lu_A.solve(b)
        u0 = x0[: self.nu]

        P_Nn1 = np.zeros(self.nla_N, dtype=float)
        P_Fn1 = np.zeros(self.nla_F, dtype=float)

        converged = True
        error = 0
        j = 0

        # identify active contacts
        g_Nn12 = self.system.g_N(tn12, qn12)
        # self.I_N = np.logical_or(g_Nn12 < 0, np.isclose(g_Nn12, np.zeros(self.system.nla_N)))
        self.I_N = g_Nn12 <= 0

        # only enter fixed-point loop if any contact is active
        if np.any(self.I_N):
            # note: we use csc_array for efficient column slicing later,
            # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_array.html#scipy.sparse.csc_array
            self.W_N = self.system.W_N(tn12, qn12, format="csc")
            self.W_F = self.system.W_F(tn12, qn12, format="csc")

            # evaluate constant xi_N and xi_F parts
            e_N = self.system.e_N
            e_F = self.system.e_F
            chi_N = self.system.g_N_dot(tn12, qn12, np.zeros_like(un))
            chi_F = self.system.gamma_F(tn12, qn12, np.zeros_like(un))
            self.xi_N0 = e_N * (self.W_N.T @ un) + (1 + e_N) * chi_N
            self.xi_F0 = e_F * (self.W_F.T @ un) + (1 + e_F) * chi_F

            # identify active tangent contacts based on active normal contacts and
            # NF-connectivity lists
            self.I_F = compute_I_F(self.I_N, self.system.NF_connectivity)

            # compute new estimates for prox parameters and get friction coefficient
            self.prox_r_N = estimate_prox_parameter(
                self.options.prox_scaling, self.W_N, M
            )
            self.prox_r_F = estimate_prox_parameter(
                self.options.prox_scaling, self.W_F, M
            )

            # warm start
            P_N = self.P_Nn.copy()
            P_F = self.P_Fn.copy()
            for j in range(self.options.fixed_point_max_iter):
                # project percussions
                P_N, P_F = self.prox(u0, P_N, P_F)

                # update rhs
                bb = b.copy()
                bb[: self.nu] += (
                    self.W_N[:, self.I_N] @ P_N[self.I_N]
                    + self.W_F[:, self.I_F] @ P_F[self.I_F]
                )

                # compute new velocities
                x = lu_A.solve(bb)
                u = x[: self.nu]

                # convergence in velocities
                diff = u - u0

                # # error measure, see Hairer1993, Section II.4
                # sc = (
                #     self.options.fixed_point_atol
                #     + np.maximum(np.abs(u), np.abs(u0))
                #     * self.options.fixed_point_rtol
                # )
                # error = np.linalg.norm(diff / sc) / sc.size**0.5
                # converged = error < 1.0

                error = np.max(np.abs(diff))
                converged = error < self.options.fixed_point_atol

                if converged:
                    P_Nn1[self.I_N] = P_N[self.I_N]
                    P_Fn1[self.I_F] = P_F[self.I_F]
                    break

                u0 = u.copy()
        else:
            x = x0

        un1, P_gn1, P_gamman1 = np.array_split(x, self.split_x)

        # second half step
        qn1 = qn12 + 0.5 * dt * self.system.q_dot(tn12, qn12, un1)

        return (
            (converged, j, error),
            tn1,
            qn1,
            un1,
            P_gn1,
            P_gamman1,
            la_c,
            P_Nn1,
            P_Fn1,
        )

    def solve(self):
        # lists storing output variables
        q = [self.qn]
        u = [self.un]
        P_g = [self.P_gn]
        P_gamma = [self.P_gamman]
        la_c = [self.la_c0]
        P_N = [self.P_Nn]
        P_F = [self.P_Fn]

        nfrac = 100
        pbar = tqdm(self.t[1:], leave=True, mininterval=0.5, miniters=nfrac)
        for _ in pbar:
            (
                (converged, j, error),
                tn1,
                qn1,
                un1,
                P_gn1,
                P_gamman1,
                la_cn1,
                P_Nn1,
                P_Fn1,
            ) = self.step()
            pbar.set_description(
                f"t: {tn1:0.2e}; fixed-point iterations: {j+1}; error: {error:.3e}"
            )
            if not converged:
                if self.options.continue_with_unconverged:
                    print(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )
                else:
                    raise RuntimeError(
                        f"fixed-point iteration not converged after {j+1} iterations with error: {error:.5e}"
                    )

            qn1, un1 = self.system.step_callback(tn1, qn1, un1)

            q.append(qn1)
            u.append(un1)
            P_g.append(P_gn1)
            P_gamma.append(P_gamman1)
            la_c.append(la_cn1)
            P_N.append(P_Nn1)
            P_F.append(P_Fn1)

            # update local variables for accepted time step
            (
                self.tn,
                self.qn,
                self.un,
                self.P_gn,
                self.P_gamman,
                self.P_Nn,
                self.P_Fn,
            ) = (tn1, qn1, un1, P_gn1, P_gamman1, P_Nn1, P_Fn1)

        return Solution(
            self.system,
            t=np.array(self.t),
            q=np.array(q),
            u=np.array(u),
            la_g=np.array(P_g) / self.dt,
            la_gamma=np.array(P_gamma) / self.dt,
            la_c=np.array(la_c),
            la_N=np.array(P_N) / self.dt,
            la_F=np.array(P_F) / self.dt,
            P_g=np.array(P_g),
            P_gamma=np.array(P_gamma),
            P_N=np.array(P_N),
            P_F=np.array(P_F),
        )
