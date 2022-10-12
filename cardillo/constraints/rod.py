import numpy as np


class Rod:
    def __init__(
        self,
        subsystem1,
        subsystem2,
        frame_ID1=np.zeros(3),
        frame_ID2=np.zeros(3),
        K_r_SP1=np.zeros(3),
        K_r_SP2=np.zeros(3),
        la_g0=None,
    ):
        self.nla_g = 1
        self.la_g0 = np.zeros(self.nla_g) if la_g0 is None else la_g0
        self.subsystem1 = subsystem1
        self.frame_ID1 = frame_ID1
        self.K_r_SP1 = K_r_SP1

        self.subsystem2 = subsystem2
        self.frame_ID2 = frame_ID2
        self.K_r_SP2 = K_r_SP2

    def assembler_callback(self):
        qDOF1 = self.subsystem1.qDOF
        qDOF2 = self.subsystem2.qDOF
        local_qDOF1 = self.subsystem1.local_qDOF_P(self.frame_ID1)
        local_qDOF2 = self.subsystem2.local_qDOF_P(self.frame_ID2)
        self.qDOF = np.concatenate((qDOF1[local_qDOF1], qDOF2[local_qDOF2]))
        self.__nq1 = nq1 = len(local_qDOF1)
        self.__nq2 = len(local_qDOF2)
        self.__nq = self.__nq1 + self.__nq2

        uDOF1 = self.subsystem1.uDOF
        uDOF2 = self.subsystem2.uDOF
        local_uDOF1 = self.subsystem1.local_uDOF_P(self.frame_ID1)
        local_uDOF2 = self.subsystem2.local_uDOF_P(self.frame_ID2)
        self.uDOF = np.concatenate((uDOF1[local_uDOF1], uDOF2[local_uDOF2]))
        self.__nu1 = nu1 = len(local_uDOF1)
        self.__nu2 = len(local_uDOF2)
        self.__nu = self.__nu1 + self.__nu2

        self.r_OP1 = lambda t, q: self.subsystem1.r_OP(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.r_OP1_q = lambda t, q: self.subsystem1.r_OP_q(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.v_P1 = lambda t, q, u: self.subsystem1.v_P(
            t, q[:nq1], u[:nu1], self.frame_ID1, self.K_r_SP1
        )
        self.a_P1 = lambda t, q, u, u_dot: self.subsystem1.a_P(
            t, q[:nq1], u[:nu1], u_dot[:nu1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1 = lambda t, q: self.subsystem1.J_P(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )
        self.J_P1_q = lambda t, q: self.subsystem1.J_P_q(
            t, q[:nq1], self.frame_ID1, self.K_r_SP1
        )

        self.r_OP2 = lambda t, q: self.subsystem2.r_OP(
            t, q[nq1:], self.frame_ID2, self.K_r_SP2
        )
        self.r_OP2_q = lambda t, q: self.subsystem2.r_OP_q(
            t, q[nq1:], self.frame_ID2, self.K_r_SP2
        )
        self.v_P2 = lambda t, q, u: self.subsystem2.v_P(
            t, q[nq1:], u[nu1:], self.frame_ID2, self.K_r_SP2
        )
        self.a_P2 = lambda t, q, u, u_dot: self.subsystem2.a_P(
            t, q[nq1:], u[nu1:], u_dot[nu1:], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2 = lambda t, q: self.subsystem2.J_P(
            t, q[nq1:], self.frame_ID2, self.K_r_SP2
        )
        self.J_P2_q = lambda t, q: self.subsystem2.J_P_q(
            t, q[nq1:], self.frame_ID2, self.K_r_SP2
        )

        r_OP10 = self.subsystem1.r_OP(
            self.subsystem1.t0, self.subsystem1.q0[qDOF1], self.frame_ID1, self.K_r_SP1
        )
        r_OP20 = self.subsystem2.r_OP(
            self.subsystem2.t0, self.subsystem2.q0[qDOF2], self.frame_ID2, self.K_r_SP2
        )
        self.dist = np.linalg.norm(r_OP20 - r_OP10)
        if self.dist < 1e-6:
            raise ValueError("Distance in rod is close to zero.")

    def g(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        return (r_OP2 - r_OP1) @ (r_OP2 - r_OP1) - self.dist**2

    def g_q_dense(self, t, q):
        r_OP1 = self.r_OP1(t, q)
        r_OP2 = self.r_OP2(t, q)
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)
        return np.array([2 * (r_OP2 - r_OP1) @ np.hstack([-r_OP1_q, r_OP2_q])])

    def g_dot(self, t, q, u):
        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        # (r_OP2 - r_OP1) @ (r_OP2 - r_OP1)  - self.dist ** 2
        return 2 * r_P1P2 @ (self.v_P2(t, q, u) - self.v_P1(t, q, u))

    def g_dot_u(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q).T, (self.la_gDOF, self.uDOF))

    def g_ddot(self, t, q, u, u_dot):
        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        v_P1P2 = self.v_P2(t, q, u) - self.v_P1(t, q, u)
        a_P1P2 = self.a_P2(t, q, u, u_dot) - self.a_P1(t, q, u, u_dot)

        return 2 * v_P1P2 @ v_P1P2 + 2 * r_P1P2 @ a_P1P2

    def g_q(self, t, q, coo):
        coo.extend(self.g_q_dense(t, q), (self.la_gDOF, self.qDOF))

    def W_g_dense(self, t, q):
        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        return 2 * np.array([np.concatenate([-J_P1.T @ r_P1P2, J_P2.T @ r_P1P2])]).T

    def W_g(self, t, q, coo):
        coo.extend(self.W_g_dense(t, q), (self.uDOF, self.la_gDOF))

    def Wla_g_q(self, t, q, la_g, coo):
        nq1 = self.__nq1
        nu1 = self.__nu1
        r_P1P2 = self.r_OP2(t, q) - self.r_OP1(t, q)
        r_OP1_q = self.r_OP1_q(t, q)
        r_OP2_q = self.r_OP2_q(t, q)
        J_P1 = self.J_P1(t, q)
        J_P2 = self.J_P2(t, q)
        J_P1_q = self.J_P1_q(t, q)
        J_P2_q = self.J_P2_q(t, q)

        # dense blocks
        dense = np.zeros((self.__nu, self.__nq))
        dense[:nu1, :nq1] = J_P1.T @ r_OP1_q + np.einsum("i,ijk->jk", -r_P1P2, J_P1_q)
        dense[:nu1, nq1:] = -J_P1.T @ r_OP2_q
        dense[nu1:, :nq1] = -J_P2.T @ r_OP1_q
        dense[nu1:, nq1:] = J_P2.T @ r_OP2_q - np.einsum("i,ijk->jk", r_P1P2, J_P2_q)

        coo.extend(2 * la_g[0] * dense, (self.uDOF, self.qDOF))
