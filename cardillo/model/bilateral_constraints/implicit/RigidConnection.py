from cardillo.math import approx_fprime
import numpy as np
from cardillo.math.algebra import cross3, ax2skew


class RigidConnection:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        la_g0=None,
    ):
        self.nla_g = 6
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF_P(self.frame_ID1)
        qDOF2 = self.subsystem2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate(
            [self.subsystem1.qDOF[qDOF1], self.subsystem2.qDOF[qDOF2]]
        )
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2

        uDOF1 = self.subsystem1.uDOF_P(self.frame_ID1)
        uDOF2 = self.subsystem2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate(
            [self.subsystem1.uDOF[uDOF1], self.subsystem2.uDOF[uDOF2]]
        )
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

        A_IK1 = self.subsystem1.A_IK(
            self.subsystem1.t0, self.subsystem1.q0[qDOF1], frame_ID=self.frame_ID1
        )
        A_IK2 = self.subsystem2.A_IK(
            self.subsystem2.t0, self.subsystem2.q0[qDOF2], frame_ID=self.frame_ID2
        )

        r_OS1 = self.subsystem1.r_OP(
            self.subsystem1.t0,
            self.subsystem1.q0[qDOF1],
            frame_ID=self.frame_ID1,
            K_r_SP=np.zeros(3),
        )
        r_OS2 = self.subsystem2.r_OP(
            self.subsystem2.t0,
            self.subsystem2.q0[qDOF2],
            frame_ID=self.frame_ID2,
            K_r_SP=np.zeros(3),
        )

        # use orientation and position of first subsystem
        A_IB0 = A_IK1
        r_OB0 = r_OS1

        # transformation from subsystem frames to B-frame
        A_K1B = A_IK1.T @ A_IB0
        A_K2B = A_IK2.T @ A_IB0

        # subsystem-fixed vectors from subsystem frame origins to B
        K1_r_S1B = A_IK1.T @ (r_OB0 - r_OS1)
        K2_r_S2B = A_IK2.T @ (r_OB0 - r_OS2)

        # auxiliary functions for subsystem 1
        self.r_OB1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.r_OB1_q1 = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.v_B1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.v_B1_q1 = lambda t, q, u: self.subsystem1.v_P_q(
            t, q[:nq1], u[:nu1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.a_B1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.a_B1_q1 = lambda t, q, u, u_dot: self.subsystem1.a_P_q(
            t, q[:nq1], u[:nu1], u_dot[:nu1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.a_B1_u1 = lambda t, q, u, u_dot: self.subsystem1.a_P_u(
            t, q[:nq1], u[:nu1], u_dot[:nu1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.J_B1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.J_B1_q1 = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], frame_ID=self.frame_ID1, K_r_SP=K1_r_S1B
        )
        self.A_IB1 = (
            lambda t, q: self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1)
            @ A_K1B
        )
        self.A_IB1_q1 = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            A_K1B,
        )
        self.Omega1 = lambda t, q, u: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1)
        self.Omega1_q1 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Omega(t, q[:nq1], u[:nu1], self.frame_ID1),
        ) + np.einsum(
            "ij,jk->ik",
            self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Omega_q(t, q[:nq1], u[:nu1], self.frame_ID1),
        )
        self.Psi1 = lambda t, q, u, u_dot: self.subsystem1.A_IK(
            t, q[:nq1], self.frame_ID1
        ) @ self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1)
        self.Psi1_q1 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1),
        ) + np.einsum(
            "ij,jk->ik",
            self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi_q(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1),
        )
        self.Psi1_u1 = lambda t, q, u, u_dot: np.einsum(
            "ij,jk->ik",
            self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_Psi_u(t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1),
        )
        self.J_R1 = lambda t, q: self.subsystem1.A_IK(
            t, q[:nq1], frame_ID=self.frame_ID1
        ) @ self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1)
        self.J_R1_q1 = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem1.A_IK_q(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_J_R(t, q[:nq1], frame_ID=self.frame_ID1),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem1.A_IK(t, q[:nq1], frame_ID=self.frame_ID1),
            self.subsystem1.K_J_R_q(t, q[:nq1], frame_ID=self.frame_ID1),
        )

        # auxiliary functions for subsystem 1
        self.r_OB2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.r_OB2_q2 = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.v_B2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.v_B2_q2 = lambda t, q, u: self.subsystem2.v_P_q(
            t, q[nq1:], u[nu1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.a_B2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.a_B2_q2 = lambda t, q, u, u_dot: self.subsystem2.a_P_q(
            t, q[nq1:], u[nu1:], u_dot[nu1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.a_B2_u2 = lambda t, q, u, u_dot: self.subsystem2.a_P_u(
            t, q[nq1:], u[nu1:], u_dot[nu1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.J_B2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.J_B2_q2 = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], frame_ID=self.frame_ID2, K_r_SP=K2_r_S2B
        )
        self.A_IB2 = (
            lambda t, q: self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2)
            @ A_K2B
        )
        self.A_IB2_q2 = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            A_K2B,
        )
        self.Omega2 = lambda t, q, u: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2)
        self.Omega2_q2 = lambda t, q, u: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Omega(t, q[nq1:], u[nu1:], self.frame_ID2),
        ) + np.einsum(
            "ij,jk->ik",
            self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Omega_q(t, q[nq1:], u[nu1:], self.frame_ID2),
        )
        self.Psi2 = lambda t, q, u, u_dot: self.subsystem2.A_IK(
            t, q[nq1:], self.frame_ID2
        ) @ self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2)
        self.Psi2_q2 = lambda t, q, u, u_dot: np.einsum(
            "ijk,j->ik",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Psi(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2),
        ) + np.einsum(
            "ij,jk->ik",
            self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Psi_q(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2),
        )
        self.Psi2_u2 = lambda t, q, u, u_dot: np.einsum(
            "ij,jk->ik",
            self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_Psi_u(t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2),
        )
        self.J_R2 = lambda t, q: self.subsystem2.A_IK(
            t, q[nq1:], frame_ID=self.frame_ID2
        ) @ self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2)
        self.J_R2_q2 = lambda t, q: np.einsum(
            "ijl,jk->ikl",
            self.subsystem2.A_IK_q(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_J_R(t, q[nq1:], frame_ID=self.frame_ID2),
        ) + np.einsum(
            "ij,jkl->ikl",
            self.subsystem2.A_IK(t, q[nq1:], frame_ID=self.frame_ID2),
            self.subsystem2.K_J_R_q(t, q[nq1:], frame_ID=self.frame_ID2),
        )

    def g(self, t, q):
        r_OB1 = self.r_OB1(t, q)
        r_OB2 = self.r_OB2(t, q)
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ey1 @ ez2, ez1 @ ex2]])

    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:3, :nq1] = -self.r_OB1_q1(t, q)
        g_q[:3, nq1:] = self.r_OB2_q2(t, q)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        ey1_q = A_IB1_q1[:, 1]
        ez1_q = A_IB1_q1[:, 2]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q = A_IB2_q2[:, 0]
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]

        g_q[3, :nq1] = ey2 @ ex1_q
        g_q[3, nq1:] = ex1 @ ey2_q
        g_q[4, :nq1] = ez2 @ ey1_q
        g_q[4, nq1:] = ey1 @ ez2_q
        g_q[5, :nq1] = ex2 @ ez1_q
        g_q[5, nq1:] = ez1 @ ex2_q
        return g_q

        # g_q_num = approx_fprime(q, lambda q: self.g(t, q))
        # diff = g_q_num - g_q
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error g_q: {error}')
        # return g_q_num

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

        # np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ey1 @ ez2, ez1 @ ex2]])
        g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_dot[4] = cross3(ey1, ez2) @ Omega21
        g_dot[5] = cross3(ez1, ex2) @ Omega21
        return g_dot

    def g_dot_q_dense(self, t, q, u):
        nq1 = self.nq1
        g_dot_q = np.zeros((self.nla_g, self._nq))

        # position
        g_dot_q[:3, :nq1] = -self.v_B1_q1(t, q, u)
        g_dot_q[:3, nq1:] = self.v_B2_q2(t, q, u)

        # orientations
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q1 = A_IB1_q1[:, 0]
        ey1_q1 = A_IB1_q1[:, 1]
        ez1_q1 = A_IB1_q1[:, 2]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q2 = A_IB2_q2[:, 0]
        ey2_q2 = A_IB2_q2[:, 1]
        ez2_q2 = A_IB2_q2[:, 2]

        Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)
        Omega1_q1 = self.Omega1_q1(t, q, u)
        Omega2_q2 = self.Omega2_q2(t, q, u)

        ex1_ey2 = cross3(ex1, ey2)
        g_dot_q[3, :nq1] = ex1_ey2 @ Omega1_q1 - Omega21 @ ax2skew(ey2) @ ex1_q1
        g_dot_q[3, nq1:] = -ex1_ey2 @ Omega2_q2 + Omega21 @ ax2skew(ex1) @ ey2_q2

        ey1_ez2 = cross3(ey1, ez2)
        g_dot_q[4, :nq1] = ey1_ez2 @ Omega1_q1 - Omega21 @ ax2skew(ez2) @ ey1_q1
        g_dot_q[4, nq1:] = -ey1_ez2 @ Omega2_q2 + Omega21 @ ax2skew(ey1) @ ez2_q2

        ez1_ex2 = cross3(ez1, ex2)
        g_dot_q[5, :nq1] = ez1_ex2 @ Omega1_q1 - Omega21 @ ax2skew(ex2) @ ez1_q1
        g_dot_q[5, nq1:] = -ez1_ex2 @ Omega2_q2 + Omega21 @ ax2skew(ez1) @ ex2_q2

        return g_dot_q

        # g_dot_q_num = approx_fprime(q, lambda q: self.g_dot(t, q, u), method="3-point")
        # diff = g_dot_q_num - g_dot_q
        # error = np.linalg.norm(diff)
        # print(f"error g_dot_q: {error}")
        # return g_dot_q_num

    def g_dot_q(self, t, q, u, coo):
        coo.extend(self.g_dot_q_dense(t, q, u), (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)

        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

        # g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_ddot[:3] = self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

        # g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_ddot[3] = (
            cross3(cross3(Omega1, ex1), ey2) + cross3(ex1, cross3(Omega2, ey2))
        ) @ Omega21 + cross3(ex1, ey2) @ Psi21

        # g_dot[4] = cross3(ey1, ez2) @ Omega21
        g_ddot[4] = (
            cross3(cross3(Omega1, ey1), ez2) + cross3(ey1, cross3(Omega2, ez2))
        ) @ Omega21 + cross3(ey1, ez2) @ Psi21

        # g_dot[5] = cross3(ez1, ex2) @ Omega21
        g_ddot[5] = (
            cross3(cross3(Omega1, ez1), ex2) + cross3(ez1, cross3(Omega2, ex2))
        ) @ Omega21 + cross3(ez1, ex2) @ Psi21

        return g_ddot

    def g_ddot_q_dense(self, t, q, u, u_dot):
        print(RuntimeWarning("RigidConnection.g_ddot_q_dense is not tested yet!"))
        nq1 = self.nq1
        g_ddot_q = np.zeros((self.nla_g, self._nq))

        # position
        g_ddot_q[:3, :nq1] = -self.a_B1_q1(t, q, u, u_dot)
        g_ddot_q[:3, nq1:] = self.a_B2_q2(t, q, u, u_dot)

        # orientations
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q1 = A_IB1_q1[:, 0]
        ey1_q1 = A_IB1_q1[:, 1]
        ez1_q1 = A_IB1_q1[:, 2]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q2 = A_IB2_q2[:, 0]
        ey2_q2 = A_IB2_q2[:, 1]
        ez2_q2 = A_IB2_q2[:, 2]

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        Omega1_q1 = self.Omega1_q1(t, q, u)
        Omega2_q2 = self.Omega2_q2(t, q, u)

        Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)
        Psi1_q1 = self.Psi1_q1(t, q, u, u_dot)
        Psi2_q2 = self.Psi2_q2(t, q, u, u_dot)

        ex1_ey2 = cross3(ex1, ey2)
        Omega1_ex1 = cross3(Omega1, ex1)
        Omega2_ey2 = cross3(Omega2, ey2)
        tmp = cross3(Omega1_ex1, ey2) - cross3(Omega2_ey2, ex1)
        g_ddot_q[3, :nq1] = (
            -Omega21
            @ (
                ax2skew(ey2) @ (ax2skew(Omega1) @ ex1_q1 - ax2skew(ex1) @ Omega1_q1)
                + ax2skew(Omega2_ey2) @ ex1_q1
            )
            + tmp @ Omega1_q1
            + ex1_ey2 @ Psi1_q1
            - Psi21 @ ax2skew(ey2) @ ex1_q1
        )
        g_ddot_q[3, nq1:] = (
            Omega21
            @ (
                ax2skew(ex1) @ (ax2skew(Omega2) @ ey2_q2 - ax2skew(ey2) @ Omega2_q2)
                + ax2skew(Omega1_ex1) @ ey2_q2
            )
            - tmp @ Omega2_q2
            - ex1_ey2 @ Psi2_q2
            + Psi21 @ ax2skew(ex1) @ ey2_q2
        )

        ey1_ez2 = cross3(ey1, ez2)
        Omega1_ey1 = cross3(Omega1, ey1)
        Omega2_ez2 = cross3(Omega2, ez2)
        tmp = cross3(Omega1_ey1, ez2) - cross3(Omega2_ez2, ey1)
        g_ddot_q[4, :nq1] = (
            -Omega21
            @ (
                ax2skew(ez2) @ (ax2skew(Omega1) @ ey1_q1 - ax2skew(ey1) @ Omega1_q1)
                + ax2skew(Omega2_ez2) @ ey1_q1
            )
            + tmp @ Omega1_q1
            + ey1_ez2 @ Psi1_q1
            - Psi21 @ ax2skew(ez2) @ ey1_q1
        )
        g_ddot_q[4, nq1:] = (
            Omega21
            @ (
                ax2skew(ey1) @ (ax2skew(Omega2) @ ez2_q2 - ax2skew(ez2) @ Omega2_q2)
                + ax2skew(Omega1_ey1) @ ez2_q2
            )
            - tmp @ Omega2_q2
            - ey1_ez2 @ Psi2_q2
            + Psi21 @ ax2skew(ey1) @ ez2_q2
        )

        ez1_ex2 = cross3(ez1, ex2)
        Omega1_ez1 = cross3(Omega1, ez1)
        Omega2_ex2 = cross3(Omega2, ex2)
        tmp = cross3(Omega1_ez1, ex2) - cross3(Omega2_ex2, ez1)
        g_ddot_q[5, :nq1] = (
            -Omega21
            @ (
                ax2skew(ex2) @ (ax2skew(Omega1) @ ez1_q1 - ax2skew(ez1) @ Omega1_q1)
                + ax2skew(Omega2_ex2) @ ez1_q1
            )
            + tmp @ Omega1_q1
            + ez1_ex2 @ Psi1_q1
            - Psi21 @ ax2skew(ex2) @ ez1_q1
        )
        g_ddot_q[5, nq1:] = (
            Omega21
            @ (
                ax2skew(ez1) @ (ax2skew(Omega2) @ ex2_q2 - ax2skew(ex2) @ Omega2_q2)
                + ax2skew(Omega1_ez1) @ ex2_q2
            )
            - tmp @ Omega2_q2
            - ez1_ex2 @ Psi2_q2
            + Psi21 @ ax2skew(ez1) @ ex2_q2
        )

        return g_ddot_q

        # g_ddot_q_num = approx_fprime(
        #     q, lambda q: self.g_ddot(t, q, u, u_dot), method="3-point"
        # )
        # diff = g_ddot_q_num - g_ddot_q
        # error = np.linalg.norm(diff)
        # # error = np.linalg.norm(diff[:3])
        # # error = np.linalg.norm(diff[3])
        # # error = np.linalg.norm(diff[4])
        # # error = np.linalg.norm(diff[5])
        # print(f"error g_ddot_q: {error}")
        # return g_ddot_q_num

    def g_ddot_q(self, t, q, u, u_dot, coo):
        coo.extend(self.g_ddot_q_dense(t, q, u, u_dot), (self.la_gDOF, self.qDOF))

    def g_ddot_u_dense(self, t, q, u, u_dot):
        print(RuntimeWarning("RigidConnection.g_ddot_u_dense is not tested yet!"))
        nu1 = self.nu1
        g_ddot_u = np.zeros((self.nla_g, self._nq))

        # position
        g_ddot_u[:3, :nu1] = -self.a_B1_u1(t, q, u, u_dot)
        g_ddot_u[:3, nu1:] = self.a_B2_u2(t, q, u, u_dot)

        # orientations
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)

        Psi1_u1 = self.Psi1_u1(t, q, u, u_dot)
        Psi2_u2 = self.Psi2_u2(t, q, u, u_dot)

        ex1_ey2 = cross3(ex1, ey2)
        Omega1_ex1 = cross3(Omega1, ex1)
        Omega2_ey2 = cross3(Omega2, ey2)
        tmp = (
            cross3(Omega1_ex1, ey2)
            - cross3(Omega2_ey2, ex1)
            + Omega21 @ ax2skew(ex1) @ ax2skew(ey2)
        )
        g_ddot_u[3, :nu1] = +tmp @ J_R1 + ex1_ey2 @ Psi1_u1
        g_ddot_u[3, nu1:] = -tmp @ J_R2 - ex1_ey2 @ Psi2_u2

        ey1_ez2 = cross3(ey1, ez2)
        Omega1_ey1 = cross3(Omega1, ey1)
        Omega2_ez2 = cross3(Omega2, ez2)
        tmp = (
            cross3(Omega1_ey1, ez2)
            - cross3(Omega2_ez2, ey1)
            + Omega21 @ ax2skew(ey1) @ ax2skew(ez2)
        )
        g_ddot_u[4, :nu1] = +tmp @ J_R1 + ey1_ez2 @ Psi1_u1
        g_ddot_u[4, nu1:] = -tmp @ J_R2 - ey1_ez2 @ Psi2_u2

        ez1_ex2 = cross3(ez1, ex2)
        Omega1_ez1 = cross3(Omega1, ez1)
        Omega2_ex2 = cross3(Omega2, ex2)
        tmp = (
            cross3(Omega1_ez1, ex2)
            - cross3(Omega2_ex2, ez1)
            + Omega21 @ ax2skew(ez1) @ ax2skew(ex2)
        )
        g_ddot_u[5, :nu1] = +tmp @ J_R1 + ez1_ex2 @ Psi1_u1
        g_ddot_u[5, nu1:] = -tmp @ J_R2 - ez1_ex2 @ Psi2_u2

        return g_ddot_u

        # g_ddot_u_num = approx_fprime(
        #     u, lambda u: self.g_ddot(t, q, u, u_dot), method="3-point"
        # )
        # diff = g_ddot_u_num - g_ddot_u
        # error = np.linalg.norm(diff)
        # # error = np.linalg.norm(diff[:3])
        # # error = np.linalg.norm(diff[3])
        # # error = np.linalg.norm(diff[4])
        # # error = np.linalg.norm(diff[5])
        # print(f"error g_ddot_u: {error}")
        # return g_ddot_u_num

    def g_ddot_u(self, t, q, u, u_dot, coo):
        coo.extend(self.g_ddot_u_dense(t, q, u, u_dot), (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        nu1 = self.nu1
        W_g = np.zeros((self._nu, self.nla_g))

        # position
        J_B1 = self.J_B1(t, q)
        J_B2 = self.J_B2(t, q)
        W_g[:nu1, :3] = -J_B1.T
        W_g[nu1:, :3] = J_B2.T

        # orientations
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

        W_g[:, 3] = cross3(ex1, ey2) @ J
        W_g[:, 4] = cross3(ey1, ez2) @ J
        W_g[:, 5] = cross3(ez1, ex2) @ J
        return W_g

        # W_g_num = approx_fprime(np.zeros(self._nu), lambda u: self.g_dot(t, q, u)).T
        # diff = W_g_num - W_g
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error W_g: {error}')
        # return W_g_num

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.nq1
        nu1 = self.nu1
        dense = np.zeros((self._nu, self._nq))

        # position
        J_B1_q = self.J_B1_q1(t, q)
        J_B2_q = self.J_B2_q2(t, q)
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", -la_g[:3], J_B1_q)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[:3], J_B2_q)

        # angular velocity
        ex1, ey1, ez1 = self.A_IB1(t, q).T
        ex2, ey2, ez2 = self.A_IB2(t, q).T
        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        ey1_q = A_IB1_q1[:, 1]
        ez1_q = A_IB1_q1[:, 2]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ex2_q = A_IB2_q2[:, 0]
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q1(t, q)
        J_R2_q = self.J_R2_q2(t, q)

        # W_g[:nu1, 3] la_g[3]= cross3(ex1, ey2) @ J_R1 * la_g[3]
        # W_g[nu1:, 3] la_g[3]= - cross3(ex1, ey2) @ J_R2 * la_g[3]
        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[3] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[3] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[3] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[3] * n_q2, J_R2
        )

        # W_g[:nu1, 4] la_g[4]= cross3(ey1, ez2) @ J_R1 * la_g[4]
        # W_g[nu1:, 4] la_g[4]= - cross3(ey1, ez2) @ J_R2 * la_g[4]
        n = cross3(ey1, ez2)
        n_q1 = -ax2skew(ez2) @ ey1_q
        n_q2 = ax2skew(ey1) @ ez2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[4] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[4] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[4] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[4] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[4] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[4] * n_q2, J_R2
        )

        # W_g[:nu1, 5] la_g[5]= cross3(ez1, ex2) @ J_R1 * la_g[5]
        # W_g[nu1:, 5] la_g[5]= - cross3(ez1, ex2) @ J_R2 * la_g[5]
        n = cross3(ez1, ex2)
        n_q1 = -ax2skew(ex2) @ ez1_q
        n_q2 = ax2skew(ez1) @ ex2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[5] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[5] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[5] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[5] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[5] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[5] * n_q2, J_R2
        )

        coo.extend(dense, (self.uDOF, self.qDOF))

        # W_g_q = approx_fprime(q, lambda q: self.W_g_dense(t, q))
        # dense_num = np.einsum('ijk,j->ik', W_g_q, la_g)
        # diff = dense_num - dense
        # diff_error = diff
        # error = np.linalg.norm(diff_error)
        # print(f'error Wla_g_q: {error}')
        # coo.extend(dense_num, (self.uDOF, self.qDOF))


class RigidConnectionCable(RigidConnection):
    """Rigid connection for cable elements. The first director of subsystem1
    is constraint such that it is orthogonal to the second and third director
    of subsystem 2, see g(t, q).

    Note:
    -----
    subsystem1 has to be the cable element!
    """

    # def __init__(self, *args, **kwargs):
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        la_g0=None,
    ):
        super().__init__(
            subsystem1,
            subsystem2,
            frame_ID1,
            frame_ID2,
            la_g0,
        )

        # overwrite some quantities of the base class
        self.nla_g = 5
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

    def g(self, t, q):
        r_OB1 = self.r_OB1(t, q)
        r_OB2 = self.r_OB2(t, q)
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])

    def g_q_dense(self, t, q):
        # return approx_fprime(q, lambda q: self.g(t, q))
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:3, :nq1] = -self.r_OB1_q1(t, q)
        g_q[:3, nq1:] = self.r_OB2_q2(t, q)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ey2_q = A_IB1_q1[:, 1]
        ez2_q = A_IB1_q1[:, 2]

        # return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])
        g_q[3, :nq1] = ey2 @ ex1_q
        g_q[3, nq1:] = ex1 @ ey2_q
        g_q[4, :nq1] = ez2 @ ex1_q
        g_q[4, nq1:] = ex1 @ ez2_q
        return g_q

    def g_dot(self, t, q, u):
        g_dot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        Omega21 = self.Omega1(t, q, u) - self.Omega2(t, q, u)

        # return np.concatenate([r_OB2 - r_OB1, [ex1 @ ey2, ex1 @ ez2]])
        g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_dot[4] = cross3(ex1, ez2) @ Omega21
        return g_dot

    # TODO:
    def g_dot_q(self, t, q, u, coo):
        dense = approx_fprime(q, lambda q: self.g_dot(t, q, u), method="3-point")
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        g_ddot = np.zeros(self.nla_g)

        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T

        Omega1 = self.Omega1(t, q, u)
        Omega2 = self.Omega2(t, q, u)
        Omega21 = Omega1 - Omega2
        Psi21 = self.Psi1(t, q, u, u_dot) - self.Psi2(t, q, u, u_dot)

        # g_dot[:3] = self.v_B2(t, q, u) - self.v_B1(t, q, u)
        g_ddot[:3] = self.a_B2(t, q, u, u_dot) - self.a_B1(t, q, u, u_dot)

        # g_dot[3] = cross3(ex1, ey2) @ Omega21
        g_ddot[3] = (
            cross3(cross3(Omega1, ex1), ey2) @ Omega21
            + cross3(ex1, cross3(Omega2, ey2)) @ Omega21
            + cross3(ex1, ey2) @ Psi21
        )

        # g_dot[4] = cross3(ex1, ez2) @ Omega21
        g_ddot[4] = (
            cross3(cross3(Omega1, ex1), ez2) @ Omega21
            + cross3(ex1, cross3(Omega2, ez2)) @ Omega21
            + cross3(ex1, ez2) @ Psi21
        )

        return g_ddot

    def g_ddot_q(self, t, q, u, u_dot, coo):
        dense = approx_fprime(
            q, lambda q: self.g_ddot(t, q, u, u_dot), method="3-point"
        )
        coo.extend(dense, (self.la_gDOF, self.qDOF))

    def g_ddot_u(self, t, q, u, u_dot, coo):
        dense = approx_fprime(
            u, lambda u: self.g_ddot(t, q, u, u_dot), method="3-point"
        )
        coo.extend(dense, (self.la_gDOF, self.uDOF))

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        # return approx_fprime(np.zeros_like(q), lambda u: self.g_dot(t, q, u)).T
        nu1 = self.nu1
        W_g = np.zeros((self._nu, self.nla_g))

        # position
        J_B1 = self.J_B1(t, q)
        J_B2 = self.J_B2(t, q)
        W_g[:nu1, :3] = -J_B1.T
        W_g[nu1:, :3] = J_B2.T

        # orientations
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        J = np.hstack([self.J_R1(t, q), -self.J_R2(t, q)])

        W_g[:, 3] = cross3(ex1, ey2) @ J
        W_g[:, 4] = cross3(ex1, ez2) @ J
        return W_g

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.nq1
        nu1 = self.nu1
        dense = np.zeros((self._nu, self._nq))

        # position
        J_B1_q = self.J_B1_q1(t, q)
        J_B2_q = self.J_B2_q2(t, q)
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", -la_g[:3], J_B1_q)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", la_g[:3], J_B2_q)

        # angular velocity
        ex1, _, _ = self.A_IB1(t, q).T
        _, ey2, ez2 = self.A_IB2(t, q).T
        A_IB1_q1 = self.A_IB1_q1(t, q)
        ex1_q = A_IB1_q1[:, 0]
        A_IB2_q2 = self.A_IB2_q2(t, q)
        ey2_q = A_IB2_q2[:, 1]
        ez2_q = A_IB2_q2[:, 2]
        J_R1 = self.J_R1(t, q)
        J_R2 = self.J_R2(t, q)
        J_R1_q = self.J_R1_q1(t, q)
        J_R2_q = self.J_R2_q2(t, q)

        # W_g[:nu1, 3] la_g[3] = cross3(ex1, ey2) @ J_R1 * la_g[3]
        # W_g[nu1:, 3] la_g[3] = - cross3(ex1, ey2) @ J_R2 * la_g[3]
        n = cross3(ex1, ey2)
        n_q1 = -ax2skew(ey2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ey2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[3] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[3] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[3] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[3] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[3] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[3] * n_q2, J_R2
        )

        # W_g[:, 4] = cross3(ex1, ez2) @ J
        # W_g[:nu1, 4] la_g[4] = cross3(ex1, ez2) @ J_R1 * la_g[4]
        # W_g[nu1:, 4] la_g[4] = -cross3(ex1, ez2) @ J_R2 * la_g[4]
        # # W_g[:nu1, 4] la_g[4] = cross3(ey1, ez2) @ J_R1 * la_g[4]
        # # W_g[nu1:, 4] la_g[4] = - cross3(ey1, ez2) @ J_R2 * la_g[4]
        n = cross3(ex1, ez2)
        n_q1 = -ax2skew(ez2) @ ex1_q
        n_q2 = ax2skew(ex1) @ ez2_q
        dense[:nu1, :nq1] += np.einsum("i,ijk->jk", la_g[4] * n, J_R1_q) + np.einsum(
            "ij,ik->kj", la_g[4] * n_q1, J_R1
        )
        dense[:nu1, nq1:] += np.einsum("ij,ik->kj", la_g[4] * n_q2, J_R1)
        dense[nu1:, :nq1] += np.einsum("ij,ik->kj", -la_g[4] * n_q1, J_R2)
        dense[nu1:, nq1:] += np.einsum("i,ijk->jk", -la_g[4] * n, J_R2_q) + np.einsum(
            "ij,ik->kj", -la_g[4] * n_q2, J_R2
        )

        coo.extend(dense, (self.uDOF, self.qDOF))


from cardillo.discretization.B_spline import B_spline_basis1D


class Rigid_beam_beam_connection2D:
    def __init__(self, beam1, beam2, la_g0=None):
        # rigid connection between to consecutive beams. End of beam1 is connected to start of beam2.
        self.nla_g = 3
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0

        self.beam1 = beam1
        self.beam2 = beam2

        self.frame_ID1 = (1,)
        self.frame_ID2 = (0,)

        N, N_xi = B_spline_basis1D(
            beam1.polynomial_degree, 1, beam1.knot_vector.data, 1
        ).T
        self.beam1_N = self.stack_shapefunctions(N, beam1.nq_el)
        self.beam1_N_xi = self.stack_shapefunctions(N_xi, beam1.nq_el)

        N, N_xi = B_spline_basis1D(
            beam2.polynomial_degree, 1, beam2.knot_vector.data, 0
        ).T
        self.beam2_N = self.stack_shapefunctions(N, beam2.nq_el)
        self.beam2_N_xi_perp = self.stack_shapefunctions_perp(N_xi, beam2.nq_el)

    def assembler_callback(self):
        qDOF1 = self.beam1.qDOF_P(self.frame_ID1)
        qDOF2 = self.beam2.qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate([self.beam1.qDOF[qDOF1], self.beam2.qDOF[qDOF2]])
        self.nq1 = nq1 = len(qDOF1)
        self.nq2 = len(qDOF2)
        self._nq = self.nq1 + self.nq2

        uDOF1 = self.beam1.uDOF_P(self.frame_ID1)
        uDOF2 = self.beam2.uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate([self.beam1.uDOF[uDOF1], self.beam2.uDOF[uDOF2]])
        self.nu1 = nu1 = len(uDOF1)
        self.nu2 = len(uDOF2)
        self._nu = self.nu1 + self.nu2

    def g(self, t, q):
        nq1 = self.nq1
        r_OP1 = self.beam1_N @ q[:nq1]
        r_OP2 = self.beam2_N @ q[nq1:]
        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        return np.concatenate([r_OP2 - r_OP1, [t @ n]])

    def g_q_dense(self, t, q):
        nq1 = self.nq1
        g_q = np.zeros((self.nla_g, self._nq))
        g_q[:2, :nq1] = -self.beam1_N
        g_q[:2, nq1:] = self.beam2_N

        # tangent vector beam 1
        t = self.beam1_N_xi @ q[:nq1]
        # normal vector beam 2
        n = self.beam2_N_xi_perp @ q[nq1:]

        g_q[2, :nq1] = n @ self.beam1_N_xi
        g_q[2, nq1:] = t @ self.beam2_N_xi_perp
        return g_q

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q).T, (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        # dense_num = Numerical_derivative(lambda t, q: self.g_q_dense(t, q).T @ la_g, order=2)._x(t, q)
        # [la_g[0], la_g[1]] @ (self.beam2_N - self.beam1_N) independent of q
        # [la_g[2] * self.beam1_N_xi.T @ n , la_g[2] * self.beam2_N_xi_perp.T @ t]
        nq1 = self.nq1
        nu1 = self.nu1

        dense = np.zeros((self._nu, self._nq))
        dense[:nu1, nq1:] = la_g[2] * self.beam1_N_xi.T @ self.beam2_N_xi_perp
        dense[nu1:, :nq1] = la_g[2] * self.beam2_N_xi_perp.T @ self.beam1_N_xi

        coo.extend(dense, (self.uDOF, self.qDOF))

    def stack_shapefunctions(self, N, nq_el):
        # return np.kron(np.eye(2), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, :n2] = N
        NN[1, n2:] = N
        return NN

    def stack_shapefunctions_perp(self, N, nq_el):
        # return np.kron(np.array([[0, -1], [1, 0]]), N)
        n2 = int(nq_el / 2)
        NN = np.zeros((2, 2 * n2))
        NN[0, n2:] = -N
        NN[1, :n2] = N
        return NN
