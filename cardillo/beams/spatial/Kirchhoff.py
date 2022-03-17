import numpy as np
import meshio
import os

from cardillo.utility.coo import Coo
from cardillo.discretization import uniform_knot_vector
from cardillo.discretization.B_spline import Knot_vector
from cardillo.math import (
    e1, e2, e3,
    pi,
    norm,
    cross3,
    ax2skew,
    smallest_rotation,
    rodriguez,
    rodriguez_inv,
    approx_fprime,
    sign,
)
from cardillo.discretization.mesh1D import Mesh1D


# switching_beam = True
switching_beam = False


class Kirchhoff:
    def __init__(
        self,
        material_model,
        A_rho0,  # TODO
        B_rho0,  # TODO
        C_rho0,  # TODO
        polynomial_degree_r,
        polynomial_degree_phi,
        nQP,
        nEl,
        Q,
        q0=None,
        u0=None,
    ):

        # beam properties
        self.materialModel = material_model  # material model
        self.A_rho0 = A_rho0  # line density
        self.B_rho0 = B_rho0  # first moment
        self.C_rho0 = C_rho0  # second moment

        # material model
        self.material_model = material_model

        # discretization parameters
        self.polynomial_degree_r = polynomial_degree_r  # polynomial degree
        self.polynomial_degree_phi = polynomial_degree_phi  # polynomial degree
        self.nQP = nQP  # number of quadrature points
        self.nEl = nEl  # number of elements

        self.knot_vector_r = Knot_vector(polynomial_degree_r, nEl)
        self.knot_vector_phi = Knot_vector(polynomial_degree_phi, nEl)
        self.nn_r = nn_r = nEl + polynomial_degree_r  # number of nodes
        self.nn_phi = nn_phi = nEl + polynomial_degree_phi  # number of nodes

        self.nn_el_r = nn_el_r = polynomial_degree_r + 1  # number of nodes per element
        self.nn_el_phi = nn_el_phi = (
            polynomial_degree_phi + 1
        )  # number of nodes per element

        # distinguish centerline and director meshes
        # - number of degrees of freedom per node of the centerline
        self.nq_n_r = nq_n_r = 3
        # - number of degrees of freedom per node of the additional angle phi
        self.nq_n_phi = nq_n_phi = 1

        self.mesh_r = Mesh1D(
            self.knot_vector_r, nQP, derivative_order=2, basis="B-spline", nq_n=nq_n_r
        )
        self.mesh_phi = Mesh1D(
            self.knot_vector_phi,
            nQP,
            derivative_order=1,
            basis="B-spline",
            nq_n=nq_n_phi,
        )

        self.nq_r = nq_r = nn_r * nq_n_r
        self.nq_phi = nq_phi = nn_phi * nq_n_phi
        # TODO:
        self.nq = nq_r + nq_phi  # total number of generalized coordinates
        self.nu = self.nq
        self.nq_el = (
            nn_el_r * nq_n_r + nn_el_phi * nq_n_phi
        )  # total number of generalized coordinates per element
        self.nq_el_r = nn_el_r * nq_n_r
        self.nq_el_phi = nn_el_phi * nq_n_phi

        # connectivity matrices for both meshes
        self.elDOF_r = self.mesh_r.elDOF
        self.elDOF_phi = (
            self.mesh_phi.elDOF + nq_r
        )  # offset of first field (centerline r)
        self.nodalDOF_r = (
            np.arange(self.nq_n_r * self.nn_r).reshape(self.nq_n_r, self.nn_r).T
        )
        self.nodalDOF_phi = (
            np.arange(self.nq_n_phi * self.nn_phi).reshape(self.nq_n_phi, self.nn_phi).T
            + nq_r
        )

        # A_RB for each quadrature point
        self.A_RB = np.zeros((nEl, self.nQP, 3, 3))
        for i in range(nEl):
            for j in range(self.nQP):
                self.A_RB[i, j] = np.eye(3)
        self.A_RB_changed = np.zeros((nEl, self.nQP), dtype=bool)

        # build global elDOF connectivity matrix
        self.elDOF = np.zeros((nEl, self.nq_el), dtype=int)
        for el in range(nEl):
            self.elDOF[el, : self.nq_el_r] = self.elDOF_r[el]
            self.elDOF[el, self.nq_el_r :] = self.elDOF_phi[el]

        # degrees of freedom on element level
        self.rDOF = np.arange(0, self.nq_el_r)
        self.phiDOF = np.arange(0, self.nq_el_phi) + self.nq_el_r

        # shape functions
        self.N_r = self.mesh_r.N  # shape functions
        self.N_r_xi = self.mesh_r.N_xi  # first derivative w.r.t. xi
        self.N_r_xixi = self.mesh_r.N_xixi  # second derivative w.r.t. xi
        self.N_phi = self.mesh_phi.N
        self.N_phi_xi = self.mesh_phi.N_xi

        # quadrature points
        self.qp = self.mesh_r.qp  # quadrature points
        self.qw = self.mesh_r.wp  # quadrature weights

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q
        self.q0 = Q.copy() if q0 is None else q0
        self.u0 = np.zeros(self.nu) if u0 is None else u0

        # evaluate shape functions at specific xi
        self.basis_functions_r = self.mesh_r.eval_basis
        self.basis_functions_phi = self.mesh_phi.eval_basis

        # reference generalized coordinates, initial coordinates and initial velocities
        self.Q = Q  # reference configuration
        self.q0 = Q.copy() if q0 is None else q0  # initial configuration
        self.u0 = np.zeros(self.nu) if u0 is None else u0  # initial velocities

        # precompute values of the reference configuration in order to save computation time
        self.J0 = np.zeros(
            (nEl, nQP)
        )  # stretch of the reference centerline, J in Harsch2020b (5)
        self.Kappa0 = np.zeros(
            (nEl, nQP, 3)
        )  # curvature of the reference configuration

        for el in range(nEl):
            # precompute quantities of the reference configuration
            Qe = self.Q[self.elDOF[el]]
            Qe_r = Qe[self.rDOF]
            Qe_phi = Qe[self.phiDOF]

            for i in range(nQP):
                # build matrix of shape function derivatives
                NN_r_xii = self.stack3r(self.N_r_xi[el, i])
                NN_r_xixii = self.stack3r(self.N_r_xixi[el, i])
                NN_phi_i = self.N_phi[el, i]
                NN_phi_xii = self.N_phi_xi[el, i]

                # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
                # the parameter space \xi and thus need to be transformed later
                r0_xi = NN_r_xii @ Qe_r
                r0_xixi = NN_r_xixii @ Qe_r
                J0i = norm(r0_xi)
                self.J0[el, i] = J0i

                phi0 = NN_phi_i @ Qe_phi
                phi0_xi = NN_phi_xii @ Qe_phi

                # compute derivatives w.r.t. the arc lenght parameter s
                r0_s = r0_xi / J0i
                phi0_s = phi0_xi / J0i

                # first director
                d1 = r0_s  # TODO: This is only valid for the reference configuration

                # build rotation matrices
                A = smallest_rotation(e1, d1)
                B = rodriguez(d1 * phi0)
                R0 = B @ A
                d1, d2, d3 = R0.T

                # torsional and flexural strains
                # raise NotImplementedError("Move on here: Use curvature terms of Mitterbach2020")
                self.Kappa0[el, i] = np.array(
                    [
                        phi0_s
                        + r0_xixi
                        @ cross3(d1, e1)
                        / (J0i * J0i * (1 + d1 @ e1)),  # Mitterbach2020 (2.105)
                        -(d3 @ r0_xixi) / J0i**2,
                        (d2 @ r0_xixi) / J0i**2,
                    ]
                )

    @staticmethod
    def straight_configuration(
        polynomial_degree_r,
        polynomial_degree_phi,
        nEl,
        L,
        greville_abscissae=True,
        r_OP=np.zeros(3),
        A_IK=np.eye(3),
    ):
        nn_r = polynomial_degree_r + nEl
        nn_phi = polynomial_degree_phi + nEl

        X = np.linspace(0, L, num=nn_r)
        Y = np.zeros(nn_r)
        Z = np.zeros(nn_r)
        if greville_abscissae:
            kv = uniform_knot_vector(polynomial_degree_r, nEl)
            for i in range(nn_r):
                X[i] = np.sum(kv[i + 1 : i + polynomial_degree_r + 1])
            X = X * L / polynomial_degree_r

        # extract first director and buld new rotation
        # note: This rotation does not coincide with the initial A_IK and has
        # to be corrected afterwards using the superimposed rotation with phi!
        d1, d2, d3 = A_IK.T
        A = smallest_rotation(e1, d1)

        # extract axis angle vector between first and desired rotation
        psi = rodriguez_inv(A.T @ A_IK)

        # extract rotation angle
        norm_psi = norm(psi)

        r0 = np.vstack((X, Y, Z)).T
        for i, r0i in enumerate(r0):
            X[i], Y[i], Z[i] = r_OP + A_IK @ r0i

        # TODO: How to compute initial phi?
        phi0 = norm_psi * np.ones(nn_phi)

        # assemble all reference generalized coordinates
        return np.concatenate([X, Y, Z, phi0])

    def element_number(self, xi):
        # note the elements coincide for both meshes!
        return self.knot_vector_r.element_number(xi)[0]

    def stack3r(self, N):
        nn_el = self.nn_el_r
        NN = np.zeros((3, self.nq_el_r))
        NN[0, :nn_el] = N
        NN[1, nn_el : 2 * nn_el] = N
        NN[2, 2 * nn_el :] = N
        return NN

    def assembler_callback(self):
        self.__M_coo()

    #########################################
    # equations of motion
    #########################################
    def M_el(self, el):
        Me = np.zeros((self.nq_el, self.nq_el))

        # # TODO
        # for i in range(self.nQP):
        #     # build matrix of shape function derivatives
        #     NN_r_i = self.stack3r(self.N_r[el, i])
        #     NN_di_i = self.stack3di(self.N_phi[el, i])

        #     # extract reference state variables
        #     J0i = self.J0[el, i]
        #     qwi = self.qw[el, i]
        #     factor_rr = NN_r_i.T @ NN_r_i * J0i * qwi
        #     factor_rdi = NN_r_i.T @ NN_di_i * J0i * qwi
        #     factor_dir = NN_di_i.T @ NN_r_i * J0i * qwi
        #     factor_didi = NN_di_i.T @ NN_di_i * J0i * qwi

        #     # delta r * ddot r
        #     Me[self.rDOF[:, None], self.rDOF] += self.A_rho0 * factor_rr
        #     # delta r * ddot d2
        #     Me[self.rDOF[:, None], self.d2DOF] += self.B_rho0[1] * factor_rdi
        #     # delta r * ddot d3
        #     Me[self.rDOF[:, None], self.d3DOF] += self.B_rho0[2] * factor_rdi

        #     # delta d2 * ddot r
        #     Me[self.d2DOF[:, None], self.rDOF] += self.B_rho0[1] * factor_dir
        #     Me[self.d2DOF[:, None], self.d2DOF] += self.C_rho0[1, 1] * factor_didi
        #     Me[self.d2DOF[:, None], self.d3DOF] += self.C_rho0[1, 2] * factor_didi

        #     # delta d3 * ddot r
        #     Me[self.d3DOF[:, None], self.rDOF] += self.B_rho0[2] * factor_dir
        #     Me[self.d3DOF[:, None], self.d2DOF] += self.C_rho0[2, 1] * factor_didi
        #     Me[self.d3DOF[:, None], self.d3DOF] += self.C_rho0[2, 2] * factor_didi

        return Me

    def __M_coo(self):
        self.__M = Coo((self.nu, self.nu))
        for el in range(self.nEl):
            # extract element degrees of freedom
            elDOF = self.elDOF[el]

            # sparse assemble element mass matrix
            self.__M.extend(self.M_el(el), (self.uDOF[elDOF], self.uDOF[elDOF]))

    def M(self, t, q, coo):
        coo.extend_sparse(self.__M)

    def E_pot(self, t, q):
        E = 0
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            E += self.E_pot_el(q[elDOF], el)
        return E

    def E_pot_el(self, qe, el):
        Ee = 0

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_phi = qe[self.phiDOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_r_xixii = self.stack3r(self.N_r_xixi[el, i])
            NN_phi_i = self.N_phi[el, i]
            NN_phi_xii = self.N_phi_xi[el, i]

            # extract reference state variables
            J0i = self.J0[el, i]
            Kappa0_i = self.Kappa0[el, i]

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe_r
            r_xixi = NN_r_xixii @ qe_r
            ji = norm(r_xi)

            phi = NN_phi_i @ qe_phi
            phi_xi = NN_phi_xii @ qe_phi

            # compute first director
            d1 = r_xi / ji

            # axial stretch
            lambda_ = ji / J0i

            if switching_beam:
                # check sign of inner product
                cos_theta = e1 @ d1
                sign_cos = sign(cos_theta)
            else:
                sign_cos = 1.0

            def kappa(e1, d1, phi, phi_xi, r_xixi, ji, J0i):
                # build rotation matrices
                A = smallest_rotation(e1, d1)
                B = rodriguez(d1 * phi)
                R = B @ A
                d1, d2, d3 = R.T

                # torsional and flexural strains
                Kappa_i = np.array(
                    [
                        phi_xi / J0i
                        + r_xixi
                        @ cross3(d1, e1)
                        / (J0i * ji * (1 + d1 @ e1)),  # Mitterbach2020 (2.105)
                        -(d3 @ r_xixi) / (J0i * ji),
                        (d2 @ r_xixi) / (J0i * ji),
                    ]
                )

                return A, B, Kappa_i

            def kappa_C(e1, d1, phi, phi_xi, r_xixi, ji, J0i):
                if False:
                    # with pi around e2
                    A_pi = np.array([
                        [-1, 0,  0],
                        [ 0, 1,  0],
                        [ 0, 0, -1]
                    ])
                    # fmt: on

                    A_C = smallest_rotation(-e1, d1) @ A_pi
                
                    # additional rotation by phi around d1
                    # B = rodriguez(d1 * phi)
                    B_C = rodriguez(d1 * (2 * pi - phi))
                    # B_C = rodriguez(-d1 * (2 * pi - phi))
                    # B_C = rodriguez(-d1 * (pi - phi))
                    # B_C = rodriguez(-d1 * (1 - 2 * pi / phi))

                    # final rotation and extraction of directors
                    # R = B @ A
                    R_C = B_C @ A_C
                    # d1, d2, d3 = R.T
                    # d1_C, d2_C, d3_C = R_C.T

                    # A_rel = R @ R_C.T

                    # print(f"A:\n{A}")
                    # print(f"A_C:\n{A_C}")
                    # print(f"R:\n{R}")
                    # print(f"R_C:\n{R_C}")
                    # # print(f"A.T @ A_C:\n{A.T @ A_C}")
                    # # print(f"norm(A.T @ A_C): {np.max(np.abs(A.T @ A_C - np.eye(3)))}")
                    # print(f"")

                    # compute curvatures
                    d1, d2, d3 = R_C.T
                    Kappa_i = np.array(
                        [
                            -phi_xi / J0i
                            + r_xixi
                            @ cross3(d1, -e1)
                            / (J0i * ji * (1 - d1 @ e1)),  # Mitterbach2020 (2.105)
                            -(d3 @ r_xixi) / (J0i * ji),
                            (d2 @ r_xixi) / (J0i * ji),
                        ]
                    )

                # TODO: Test singular case:
                # For this case rodriguez(-pi * n_C) yields identity but not a 
                # rotation with pi around some axis!
                d1 = -e1
                d1 = -e1 + np.random.rand(3) * 1.0e-3

                # smallest rotations
                A = smallest_rotation(e1, d1)
                e1xd1_C = cross3(-e1, d1)
                n_C = e1xd1_C / norm(e1xd1_C)
                A_pi = rodriguez(-pi * n_C)
                A_C = smallest_rotation(-e1, d1) @ A_pi
                
                # additional rotation by phi around d1
                B = rodriguez(d1 * phi)
                B_C = rodriguez(d1 * phi)

                # composition of rotations
                R = B @ A
                R_C = B_C @ A_C

                print(f"A:\n{A}")
                print(f"A_C:\n{A_C}")
                print(f"B:\n{B}")
                print(f"B_C:\n{B_C}")
                print(f"")
                assert np.allclose(A.T @ A_C, np.eye(3))
                assert np.allclose(B.T @ B_C, np.eye(3))

                raise RuntimeError("This formulation has the same singularity!")

                # extract directors
                d1, d2, d3 = R_C.T

                # compute curvatures
                # TODO: A_pi curvature is missing for kappa_1 term!
                Kappa_i = np.array(
                    [
                        -phi_xi / J0i
                        + r_xixi
                        @ cross3(d1, -e1)
                        / (J0i * ji * (1 - d1 @ e1)),  # Mitterbach2020 (2.105)
                        -(d3 @ r_xixi) / (J0i * ji),
                        (d2 @ r_xixi) / (J0i * ji),
                    ]
                )
                
                return A_C, B_C, Kappa_i


            if sign_cos >= 0:
                A, B, Kappa_i = kappa(e1, d1, phi, phi_xi, r_xixi, ji, J0i)
            else:
                # A, B, Kappa_i = kappa(e1, d1, phi, phi_xi, r_xixi, ji, J0i)
                # # print(f"A:\n{A}")
                # # print(f"B:\n{B}")
                # print(f"Kappa_i:\n{Kappa_i}")
                A_C, B_C, Kappa_i_C = kappa_C(e1, d1, phi, phi_xi, r_xixi, ji, J0i)
                # # print(f"A_C:\n{A_C}")
                # # print(f"B_C:\n{B_C}")
                # print(f"Kappa_i_C:\n{Kappa_i_C}")
                # print(f"")

                # # ensure that both formulation yield in the exactly same rotations
                # # print(f"A.T @ A_C:\n{A.T @ A_C}")
                # # print(f"B.T @ B_C:\n{B.T @ B_C}")
                # assert np.allclose(A.T @ A_C, np.eye(3))
                # assert np.allclose(B.T @ B_C, np.eye(3))

                # TODO: Compute curvature term from rodriguez(e_C * np.pi)
                # print(f"")
                Kappa_i = Kappa_i_C

                # phi_xi / J0i + r_xixi @ cross3(d1, -e1) / (J0i * ji * (1 - d1 @ e1))

            # evaluate strain energy function
            Ee += (
                self.material_model.potential(lambda_, Kappa_i, Kappa0_i)
                * J0i
                * self.qw[el, i]
            )

        return Ee

    def f_pot(self, t, q):
        f = np.zeros(self.nu)
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            f[elDOF] += self.f_pot_el(q[elDOF], el)
        return f

    def f_pot_el(self, qe, el):
        # return -approx_fprime(qe, lambda qe: self.E_pot_el(qe, el), method="2-point")
        
        fe = np.zeros(self.nq_el)

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_phi = qe[self.phiDOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_r_xixii = self.stack3r(self.N_r_xixi[el, i])
            NN_phi_i = self.N_phi[el, i]
            NN_phi_xii = self.N_phi_xi[el, i]

            # extract reference state variables
            J0i = self.J0[el, i]
            Kappa0_i = self.Kappa0[el, i]

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe_r
            r_xixi = NN_r_xixii @ qe_r
            ji = norm(r_xi)

            phi = NN_phi_i @ qe_phi
            phi_xi = NN_phi_xii @ qe_phi

            # compute first director
            d1 = r_xi / ji

            # build rotation matrices
            A = smallest_rotation(e1, d1)
            B = rodriguez(d1 * phi)
            R = B @ A
            d1, d2, d3 = R.T
        
            # compute curvatures
            kappa_1 = phi_xi / J0i + r_xixi @ cross3(d1, e1) / ((J0i * ji) * (1 + d1 @ e1)) # Mitterbach2020 (2.83)
            kappa_2 = -r_xixi @ d3 / (J0i * ji) # Mitterbach2020 (2.56)
            kappa_3 =  r_xixi @ d2 / (J0i * ji) # Mitterbach2020 (2.56)

            # axial strain
            lambda_ = norm(r_xi) / J0i

            # torsional and flexural strains
            Kappa_i = np.array([kappa_1, kappa_2, kappa_3])
            
            # evaluate contact forces and couples
            n_1 = self.material_model.n_1(lambda_, Kappa_i, Kappa0_i)
            m_i = self.material_model.m_i(lambda_, Kappa_i, Kappa0_i)

            # variation of strain measures
            delta_lambda_r = r_xi @ NN_r_xii / (J0i * ji) # Mitterbach (2.91)

            C3 = 1 / (J0i * ji**2 * (1 + d1 @ e1))
            C4 = 1 / (J0i * ji**3 * (1 + d1 @ e1)**2)
            delta_kappa_1_phi = NN_phi_xii / J0i # Mitterbach (2.99)
            delta_kappa_1_r = (C3 * cross3(e1, r_xixi) \
                                - 2 * C3 * (r_xixi @ cross3(r_xi, e1) / ji**2) * r_xi \
                                - C4 * (r_xixi @ cross3(r_xi, e1)) * e1 \
                                + C4 * (r_xixi @ cross3(r_xi, e1) * (d1 @ e1)) * d1
                                ) @ NN_r_xii \
                                + C3 * cross3(r_xi, e1) @ NN_r_xixii # Mitterbach (2.99)

            delta_kappa_2_phi = (r_xixi @ d2 / (J0i * ji)) * NN_phi_i # Mitterbach (2.123)
            delta_kappa_3_phi = (r_xixi @ d3 / (J0i * ji)) * NN_phi_i # Mitterbach (2.124)

            delta_kappa_2_r = ((r_xixi @ d3) / (J0i * ji**2) * d1 \
                                + (r_xixi @ d1) / (J0i * ji**2) * d3 \
                                + (r_xixi @ d2) / (J0i * ji**2 * (1 + d1 @ e1)) * cross3(d1, e1)
                                ) @ NN_r_xii \
                                - d3 @ NN_r_xixii / (J0i * ji) # Mitterbach (2.123)

            delta_kappa_3_r = (- (r_xixi @ d2) / (J0i * ji**2) * d1 \
                                - (r_xixi @ d1) / (J0i * ji**2) * d2 \
                                + (r_xixi @ d3) / (J0i * ji**2 * (1 + d1 @ e1)) * cross3(d1, e1)
                                ) @ NN_r_xii \
                                + d2 @ NN_r_xixii / (J0i * ji) # Mitterbach (2.124)

            fe[self.rDOF] -= (n_1 * delta_lambda_r + m_i[0] * delta_kappa_1_r + m_i[1] * delta_kappa_2_r + m_i[2] * delta_kappa_3_r) * J0i * self.qw[el, i]
            fe[self.phiDOF] -= (m_i[0] * delta_kappa_1_phi + m_i[1] * delta_kappa_2_phi + m_i[2] * delta_kappa_3_phi) * J0i * self.qw[el, i]

        return fe

    def f_pot_q(self, t, q, coo):
        for el in range(self.nEl):
            elDOF = self.elDOF[el]
            Ke = self.f_pot_q_el(q[elDOF], el)

            # sparse assemble element internal stiffness matrix
            coo.extend(Ke, (self.uDOF[elDOF], self.qDOF[elDOF]))

    def f_pot_q_el(self, qe, el):
        return approx_fprime(qe, lambda qe: self.f_pot_el(qe, el), method="2-point")

        Ke = np.zeros((self.nq_el, self.nq_el))

        # extract generalized coordinates for beam centerline and directors
        # in the current and reference configuration
        qe_r = qe[self.rDOF]
        qe_d1 = qe[self.phiDOF]
        qe_d2 = qe[self.d2DOF]
        qe_d3 = qe[self.d3DOF]

        for i in range(self.nQP):
            # build matrix of shape function derivatives
            # NN_r_i = self.stack3r(self.N_r[el, i])
            NN_di_i = self.stack3di(self.N_phi[el, i])
            NN_r_xii = self.stack3r(self.N_r_xi[el, i])
            NN_di_xii = self.stack3di(self.N_phi_xi[el, i])

            # extract reference state variables
            J0i = self.J0[el, i]
            Gamma0_i = self.lambda0[el, i]
            Kappa0_i = self.Kappa0[el, i]
            qwi = self.qw[el, i]

            # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
            # the parameter space \xi and thus need to be transformed later
            r_xi = NN_r_xii @ qe_r

            d1 = NN_di_i @ qe_d1
            d1_xi = NN_di_xii @ qe_d1

            d2 = NN_di_i @ qe_d2
            d2_xi = NN_di_xii @ qe_d2

            d3 = NN_di_i @ qe_d3
            d3_xi = NN_di_xii @ qe_d3

            # compute derivatives w.r.t. the arc lenght parameter s
            r_s = r_xi / J0i

            d1_s = d1_xi / J0i
            d2_s = d2_xi / J0i
            d3_s = d3_xi / J0i

            # build rotation matrices
            R = np.vstack((d1, d2, d3)).T

            # axial and shear strains
            Gamma_i = R.T @ r_s

            # derivative of axial and shear strains
            Gamma_j_qr = R.T @ NN_r_xii / J0i
            Gamma_1_qd1 = r_xi @ NN_di_i / J0i
            Gamma_2_qd2 = r_xi @ NN_di_i / J0i
            Gamma_3_qd3 = r_xi @ NN_di_i / J0i

            #################################################################
            # formulation of Harsch2020b
            #################################################################

            # torsional and flexural strains
            Kappa_i = np.array(
                [
                    0.5 * (d3 @ d2_s - d2 @ d3_s),
                    0.5 * (d1 @ d3_s - d3 @ d1_s),
                    0.5 * (d2 @ d1_s - d1 @ d2_s),
                ]
            )

            # derivative of torsional and flexural strains
            kappa_1_qe_d1 = np.zeros(3 * self.nn_el_phi)
            kappa_1_qe_d2 = 0.5 * (d3 @ NN_di_xii - d3_xi @ NN_di_i) / J0i
            kappa_1_qe_d3 = 0.5 * (d2_xi @ NN_di_i - d2 @ NN_di_xii) / J0i

            kappa_2_qe_d1 = 0.5 * (d3_xi @ NN_di_i - d3 @ NN_di_xii) / J0i
            kappa_2_qe_d2 = np.zeros(3 * self.nn_el_phi)
            kappa_2_qe_d3 = 0.5 * (d1 @ NN_di_xii - d1_xi @ NN_di_i) / J0i

            kappa_3_qe_d1 = 0.5 * (d2 @ NN_di_xii - d2_xi @ NN_di_i) / J0i
            kappa_3_qe_d2 = 0.5 * (d1_xi @ NN_di_i - d1 @ NN_di_xii) / J0i
            kappa_3_qe_d3 = np.zeros(3 * self.nn_el_phi)

            kappa_j_qe_d1 = np.vstack((kappa_1_qe_d1, kappa_2_qe_d1, kappa_3_qe_d1))
            kappa_j_qe_d2 = np.vstack((kappa_1_qe_d2, kappa_2_qe_d2, kappa_3_qe_d2))
            kappa_j_qe_d3 = np.vstack((kappa_1_qe_d3, kappa_2_qe_d3, kappa_3_qe_d3))

            # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            n1, n2, n3 = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            n_i_Gamma_j = self.material_model.n_i_Gamma_j(
                Gamma_i, Gamma0_i, Kappa_i, Kappa0_i
            )
            n_i_Kappa_j = self.material_model.n_i_K_j(
                Gamma_i, Gamma0_i, Kappa_i, Kappa0_i
            )
            n1_qr, n2_qr, n3_qr = n_i_Gamma_j @ Gamma_j_qr

            n1_qd1, n2_qd1, n3_qd1 = (
                np.outer(n_i_Gamma_j[0], Gamma_1_qd1) + n_i_Kappa_j @ kappa_j_qe_d1
            )
            n1_qd2, n2_qd2, n3_qd2 = (
                np.outer(n_i_Gamma_j[1], Gamma_2_qd2) + n_i_Kappa_j @ kappa_j_qe_d2
            )
            n1_qd3, n2_qd3, n3_qd3 = (
                np.outer(n_i_Gamma_j[2], Gamma_3_qd3) + n_i_Kappa_j @ kappa_j_qe_d3
            )

            m1, m2, m3 = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            m_i_Gamma_j = self.material_model.m_i_Gamma_j(
                Gamma_i, Gamma0_i, Kappa_i, Kappa0_i
            )
            m_i_Kappa_j = self.material_model.m_i_K_j(
                Gamma_i, Gamma0_i, Kappa_i, Kappa0_i
            )

            m1_qr, m2_qr, m3_qr = m_i_Gamma_j @ Gamma_j_qr
            m1_qd1, m2_qd1, m3_qd1 = (
                np.outer(m_i_Gamma_j[0], Gamma_1_qd1) + m_i_Kappa_j @ kappa_j_qe_d1
            )
            m1_qd2, m2_qd2, m3_qd2 = (
                np.outer(m_i_Gamma_j[1], Gamma_2_qd2) + m_i_Kappa_j @ kappa_j_qe_d2
            )
            m1_qd3, m2_qd3, m3_qd3 = (
                np.outer(m_i_Gamma_j[2], Gamma_3_qd3) + m_i_Kappa_j @ kappa_j_qe_d3
            )

            Ke[self.rDOF[:, None], self.rDOF] -= (
                NN_r_xii.T
                @ (np.outer(d1, n1_qr) + np.outer(d2, n2_qr) + np.outer(d3, n3_qr))
                * qwi
            )
            Ke[self.rDOF[:, None], self.phiDOF] -= (
                NN_r_xii.T
                @ (
                    np.outer(d1, n1_qd1)
                    + np.outer(d2, n2_qd1)
                    + np.outer(d3, n3_qd1)
                    + n1 * NN_di_i
                )
                * qwi
            )
            Ke[self.rDOF[:, None], self.d2DOF] -= (
                NN_r_xii.T
                @ (
                    np.outer(d1, n1_qd2)
                    + np.outer(d2, n2_qd2)
                    + np.outer(d3, n3_qd2)
                    + n2 * NN_di_i
                )
                * qwi
            )
            Ke[self.rDOF[:, None], self.d3DOF] -= (
                NN_r_xii.T
                @ (
                    np.outer(d1, n1_qd3)
                    + np.outer(d2, n2_qd3)
                    + np.outer(d3, n3_qd3)
                    + n3 * NN_di_i
                )
                * qwi
            )

            Ke[self.phiDOF[:, None], self.rDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n1_qr)
                    + n1 * NN_r_xii
                    + np.outer(0.5 * d3_xi, m2_qr)
                    - np.outer(0.5 * d2_xi, m3_qr)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d2, m3_qr) - np.outer(0.5 * d3, m2_qr))
                * qwi
            )
            Ke[self.phiDOF[:, None], self.phiDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n1_qd1)
                    + np.outer(0.5 * d3_xi, m2_qd1)
                    - np.outer(0.5 * d2_xi, m3_qd1)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d2, m3_qd1) - np.outer(0.5 * d3, m2_qd1))
                * qwi
            )
            Ke[self.phiDOF[:, None], self.d2DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n1_qd2)
                    + np.outer(0.5 * d3_xi, m2_qd2)
                    - np.outer(0.5 * d2_xi, m3_qd2)
                    - 0.5 * m3 * NN_di_xii
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d2, m3_qd2)
                    + 0.5 * m3 * NN_di_i
                    - np.outer(0.5 * d3, m2_qd2)
                )
                * qwi
            )
            Ke[self.phiDOF[:, None], self.d3DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n1_qd3)
                    + np.outer(0.5 * d3_xi, m2_qd3)
                    + 0.5 * m2 * NN_di_xii
                    - np.outer(0.5 * d2_xi, m3_qd3)
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d2, m3_qd3)
                    - np.outer(0.5 * d3, m2_qd3)
                    - 0.5 * m2 * NN_di_i
                )
                * qwi
            )

            Ke[self.d2DOF[:, None], self.rDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n2_qr)
                    + n2 * NN_r_xii
                    + np.outer(0.5 * d1_xi, m3_qr)
                    - np.outer(0.5 * d3_xi, m1_qr)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d3, m1_qr) - np.outer(0.5 * d1, m3_qr))
                * qwi
            )
            Ke[self.d2DOF[:, None], self.phiDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n2_qd1)
                    + np.outer(0.5 * d1_xi, m3_qd1)
                    + 0.5 * m3 * NN_di_xii
                    - np.outer(0.5 * d3_xi, m1_qd1)
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d3, m1_qd1)
                    - np.outer(0.5 * d1, m3_qd1)
                    - 0.5 * m3 * NN_di_i
                )
                * qwi
            )
            Ke[self.d2DOF[:, None], self.d2DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n2_qd2)
                    + np.outer(0.5 * d1_xi, m3_qd2)
                    - np.outer(0.5 * d3_xi, m1_qd2)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d3, m1_qd2) - np.outer(0.5 * d1, m3_qd2))
                * qwi
            )
            Ke[self.d2DOF[:, None], self.d3DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n2_qd3)
                    + np.outer(0.5 * d1_xi, m3_qd3)
                    - np.outer(0.5 * d3_xi, m1_qd3)
                    - 0.5 * m1 * NN_di_xii
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d3, m1_qd3)
                    + 0.5 * m1 * NN_di_i
                    - np.outer(0.5 * d1, m3_qd3)
                )
                * qwi
            )

            Ke[self.d3DOF[:, None], self.rDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n3_qr)
                    + n3 * NN_r_xii
                    + np.outer(0.5 * d2_xi, m1_qr)
                    - np.outer(0.5 * d1_xi, m2_qr)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d1, m2_qr) - np.outer(0.5 * d2, m1_qr))
                * qwi
            )
            Ke[self.d3DOF[:, None], self.phiDOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n3_qd1)
                    + np.outer(0.5 * d2_xi, m1_qd1)
                    - np.outer(0.5 * d1_xi, m2_qd1)
                    - 0.5 * m2 * NN_di_xii
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d1, m2_qd1)
                    + 0.5 * m2 * NN_di_i
                    - np.outer(0.5 * d2, m1_qd1)
                )
                * qwi
            )
            Ke[self.d3DOF[:, None], self.d2DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n3_qd2)
                    + np.outer(0.5 * d2_xi, m1_qd2)
                    + 0.5 * m1 * NN_di_xii
                    - np.outer(0.5 * d1_xi, m2_qd2)
                )
                * qwi
                + NN_di_xii.T
                @ (
                    np.outer(0.5 * d1, m2_qd2)
                    - np.outer(0.5 * d2, m1_qd2)
                    - 0.5 * m1 * NN_di_i
                )
                * qwi
            )
            Ke[self.d3DOF[:, None], self.d3DOF] -= (
                NN_di_i.T
                @ (
                    np.outer(r_xi, n3_qd3)
                    + np.outer(0.5 * d2_xi, m1_qd3)
                    - np.outer(0.5 * d1_xi, m2_qd3)
                )
                * qwi
                + NN_di_xii.T
                @ (np.outer(0.5 * d1, m2_qd3) - np.outer(0.5 * d2, m1_qd3))
                * qwi
            )

            # #################################################################
            # # alternative formulation assuming orthogonality of the directors
            # #################################################################

            # # torsional and flexural strains
            # Kappa_i = np.array([d3 @ d2_s, \
            #                     d1 @ d3_s, \
            #                     d2 @ d1_s])

            # # derivative of torsional and flexural strains
            # kappa_1_qe_d1 = np.zeros(3 * self.nn_el_di)
            # kappa_1_qe_d2 = d3 @ NN_di_xii / J0i
            # kappa_1_qe_d3 = d2_s @ NN_di_i

            # kappa_2_qe_d1 = d3_s @ NN_di_i
            # kappa_2_qe_d2 = np.zeros(3 * self.nn_el_di)
            # kappa_2_qe_d3 = d1 @ NN_di_xii / J0i

            # kappa_3_qe_d1 = d2 @ NN_di_xii / J0i
            # kappa_3_qe_d2 = d1_s @ NN_di_i
            # kappa_3_qe_d3 = np.zeros(3 * self.nn_el_di)

            # kappa_j_qe_d1 = np.vstack((kappa_1_qe_d1, kappa_2_qe_d1, kappa_3_qe_d1))
            # kappa_j_qe_d2 = np.vstack((kappa_1_qe_d2, kappa_2_qe_d2, kappa_3_qe_d2))
            # kappa_j_qe_d3 = np.vstack((kappa_1_qe_d3, kappa_2_qe_d3, kappa_3_qe_d3))

            # # compute contact forces and couples from partial derivatives of the strain energy function w.r.t. strain measures
            # n1, n2, n3 = self.material_model.n_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n_i_Gamma_i_j = self.material_model.n_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n_i_Kappa_i_j = self.material_model.n_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # n1_qr, n2_qr, n3_qr = n_i_Gamma_i_j @ Gamma_j_qr

            # n1_qd1, n2_qd1, n3_qd1 = np.outer(n_i_Gamma_i_j[0], Gamma_1_qd1) + n_i_Kappa_i_j @ kappa_j_qe_d1
            # n1_qd2, n2_qd2, n3_qd2 = np.outer(n_i_Gamma_i_j[1], Gamma_2_qd2) + n_i_Kappa_i_j @ kappa_j_qe_d2
            # n1_qd3, n2_qd3, n3_qd3 = np.outer(n_i_Gamma_i_j[2], Gamma_3_qd3) + n_i_Kappa_i_j @ kappa_j_qe_d3

            # m1, m2, m3 = self.material_model.m_i(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # m_i_Gamma_i_j = self.material_model.m_i_Gamma_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)
            # m_i_Kappa_i_j = self.material_model.m_i_Kappa_i_j(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i)

            # m1_qr, m2_qr, m3_qr = m_i_Gamma_i_j @ Gamma_j_qr
            # m1_qd1, m2_qd1, m3_qd1 = np.outer(m_i_Gamma_i_j[0], Gamma_1_qd1) + m_i_Kappa_i_j @ kappa_j_qe_d1
            # m1_qd2, m2_qd2, m3_qd2 = np.outer(m_i_Gamma_i_j[1], Gamma_2_qd2) + m_i_Kappa_i_j @ kappa_j_qe_d2
            # m1_qd3, m2_qd3, m3_qd3 = np.outer(m_i_Gamma_i_j[2], Gamma_3_qd3) + m_i_Kappa_i_j @ kappa_j_qe_d3

            # # quadrature point contribution to element stiffness matrix

            # # fe[self.rDOF] -=  NN_r_xii.T @ ( n1 * d1 + n2 * d2 + n3 * d3 ) * qwi # delta r'
            # Ke[self.rDOF[:, None], self.rDOF] -= NN_r_xii.T @ (np.outer(d1, n1_qr) + np.outer(d2, n2_qr) + np.outer(d3, n3_qr)) * qwi
            # Ke[self.rDOF[:, None], self.d1DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd1) + np.outer(d2, n2_qd1) + np.outer(d3, n3_qd1) + n1 * NN_di_i) * qwi
            # Ke[self.rDOF[:, None], self.d2DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd2) + np.outer(d2, n2_qd2) + np.outer(d3, n3_qd2) + n2 * NN_di_i) * qwi
            # Ke[self.rDOF[:, None], self.d3DOF] -= NN_r_xii.T @ (np.outer(d1, n1_qd3) + np.outer(d2, n2_qd3) + np.outer(d3, n3_qd3) + n3 * NN_di_i) * qwi

            # # fe[self.d1DOF] -= ( NN_di_i.T @ ( r_xi * n1 + m2 * d3_xi ) # delta d1 \
            # #                     + NN_di_xii.T @ ( m3 * d2 ) ) * qwi    # delta d1'
            # Ke[self.d1DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qr) + n1 * NN_r_xii + np.outer(d3_xi, m2_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d2, m3_qr) * qwi
            # Ke[self.d1DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd1) + np.outer(d3_xi, m2_qd1)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d2, m3_qd1) * qwi
            # Ke[self.d1DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd2) + np.outer(d3_xi, m2_qd2)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d2, m3_qd2) + m3 * NN_di_i) * qwi
            # Ke[self.d1DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n1_qd3) + np.outer(d3_xi, m2_qd3) + m2 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d2, m3_qd3) * qwi

            # # fe[self.d2DOF] -= ( NN_di_i.T @ ( r_xi * n2 + m3 * d1_xi ) # delta d2 \
            # #                     + NN_di_xii.T @ ( m1 * d3 ) ) * qwi    # delta d2'
            # Ke[self.d2DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qr) + n2 * NN_r_xii + np.outer(d1_xi, m3_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d3, m1_qr) * qwi
            # Ke[self.d2DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd1) + np.outer(d1_xi, m3_qd1) + m3 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d3, m1_qd1) * qwi
            # Ke[self.d2DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd2) + np.outer(d1_xi, m3_qd2)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d3, m1_qd2) * qwi
            # Ke[self.d2DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n2_qd3) + np.outer(d1_xi, m3_qd3)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d3, m1_qd3) + m1 * NN_di_i) * qwi

            # # fe[self.d3DOF] -= ( NN_di_i.T @ ( r_xi * n3 + m1 * d2_xi ) # delta d3 \
            # #                     + NN_di_xii.T @ ( m2 * d1 ) ) * qwi    # delta d3'
            # Ke[self.d3DOF[:, None], self.rDOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qr) + n3 * NN_r_xii + np.outer(d2_xi, m1_qr)) * qwi \
            #                                       + NN_di_xii.T @ np.outer(d1, m2_qr) * qwi
            # Ke[self.d3DOF[:, None], self.d1DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd1) + np.outer(d2_xi, m1_qd1)) * qwi \
            #                                        + NN_di_xii.T @ (np.outer(d1, m2_qd1) + m2 * NN_di_i) * qwi
            # Ke[self.d3DOF[:, None], self.d2DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd2) + np.outer(d2_xi, m1_qd2) + m1 * NN_di_xii) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d1, m2_qd2) * qwi
            # Ke[self.d3DOF[:, None], self.d3DOF] -= NN_di_i.T @ (np.outer(r_xi, n3_qd3) + np.outer(d2_xi, m1_qd3)) * qwi \
            #                                        + NN_di_xii.T @ np.outer(d1, m2_qd3) * qwi

        return Ke

        # Ke_num = Numerical_derivative(lambda t, qe: self.f_pot_el(qe, el), order=2)._x(0, qe)
        # diff = Ke_num - Ke
        # error = np.max(np.abs(diff))
        # print(f'max error f_pot_q_el: {error}')

        # print(f'diff[self.rDOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.rDOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.rDOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.rDOF[:, None], self.d3DOF]))}')

        # print(f'diff[self.d1DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d1DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d1DOF[:, None], self.d3DOF]))}')

        # print(f'diff[self.d2DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d2DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d2DOF[:, None], self.d3DOF]))}')

        # print(f'diff[self.d3DOF[:, None], self.rDOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.rDOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d1DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d1DOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d2DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d2DOF]))}')
        # print(f'diff[self.d3DOF[:, None], self.d3DOF]: {np.max(np.abs(diff[self.d3DOF[:, None], self.d3DOF]))}')

        # return Ke_num

    ####################
    # kinematic equation
    ####################
    def q_dot(self, t, q, u):
        return u

    def B(self, t, q, coo):
        coo.extend_diag(np.ones(self.nq), (self.qDOF, self.uDOF))

    def q_ddot(self, t, q, u, u_dot):
        return u_dot

    ####################################################
    # interactions with other bodies and the environment
    ####################################################
    # TODO: optimized implementation for boundaries
    def elDOF_P(self, frame_ID):
        xi = frame_ID[0]
        el = self.element_number(xi)
        return self.elDOF[el]

    def qDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def uDOF_P(self, frame_ID):
        return self.elDOF_P(frame_ID)

    def r_OP(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        N, _, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)
        return NN @ q[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ K_r_SP

    def r_OP_q(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        N, _, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        r_OP_q = np.zeros((3, self.nq_el))
        r_OP_q[:, self.rDOF] = NN
        r_OP_q += np.einsum("ijk,j->ik", self.A_IK_q(t, q, frame_ID=frame_ID), K_r_SP)
        return r_OP_q

    def A_IK(self, t, q, frame_ID):
        # evaluate basis functions
        _, N_xi, _ = self.basis_functions_r(frame_ID[0])
        N, _ = self.basis_functions_phi(frame_ID[0])

        # evaluate tangent and angle
        NN_xi = self.stack3r(N_xi)
        r_xi = NN_xi @ q[self.rDOF]
        phi = N @ q[self.phiDOF]

        # compute first director
        ji = norm(r_xi)
        d1 = r_xi / ji

        # build rotation matrices
        A = smallest_rotation(e1, d1)
        B = rodriguez(d1 * phi)
        return B @ A

    # TODO
    def A_IK_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.A_IK(t, q, frame_ID))

    def v_P(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        N, _, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        v_P = NN @ u[self.rDOF] + self.A_IK(t, q, frame_ID) @ cross3(
            self.K_Omega(t, q, u, frame_ID=frame_ID), K_r_SP
        )
        return v_P

    # TODO
    def v_P_q(self, t, q, u, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(
            q, lambda q: self.v_P(t, q, u, frame_ID, K_r_SP), method="2-point"
        )

    # TODO
    def J_P(self, t, q, frame_ID, K_r_SP=np.zeros(3)):
        return approx_fprime(
            np.zeros_like(q),
            lambda u: self.v_P(t, q, u, frame_ID, K_r_SP),
            method="3-point",
        )

    # TODO
    def J_P_q(self, t, q, frame_ID=None, K_r_SP=np.zeros(3)):
        return approx_fprime(
            q, lambda q: self.J_P(t, q, frame_ID, K_r_SP), method="2-point"
        )

    def a_P(self, t, q, u, u_dot, frame_ID, K_r_SP=np.zeros(3)):
        N, _ = self.basis_functions_r(frame_ID[0])
        NN = self.stack3r(N)

        K_Omega = self.K_Omega(t, q, u, frame_ID=frame_ID)
        K_Psi = self.K_Psi(t, q, u, u_dot, frame_ID=frame_ID)
        a_P = NN @ u_dot[self.rDOF] + self.A_IK(t, q, frame_ID=frame_ID) @ (
            cross3(K_Psi, K_r_SP) + cross3(K_Omega, cross3(K_Omega, K_r_SP))
        )
        return a_P

    def a_P_q(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return approx_fprime(q, lambda q: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP))

    def a_P_u(self, t, q, u, u_dot, frame_ID, K_r_SP=None):
        return approx_fprime(u, lambda u: self.a_P(t, q, u, u_dot, frame_ID, K_r_SP))

    def K_Omega(self, t, q, u, frame_ID):
        # evaluate basis functions
        _, N_xi, _ = self.basis_functions_r(frame_ID[0])
        N, _ = self.basis_functions_phi(frame_ID[0])

        # evaluate tangent and angle + their time derivatives
        NN_xi = self.stack3r(N_xi)
        r_xi = NN_xi @ q[self.rDOF]
        r_xi_dot = NN_xi @ u[self.rDOF]
        phi = N @ q[self.phiDOF]
        phi_dot = N @ u[self.phiDOF]

        # compute first director
        ji = norm(r_xi)
        d1 = r_xi / ji

        def Omega(e1, d1, phi, phi_dot, r_xi_dot, ji):
            # build rotation matrices
            A = smallest_rotation(e1, d1)
            B = rodriguez(d1 * phi)
            R = B @ A
            d1, d2, d3 = R.T

            K_Omega = np.array(
                [
                    phi_dot
                    + r_xi_dot
                    @ cross3(d1, e1)
                    / (ji * (1 + d1 @ e1)),  # Mitterbach2020 (2.105)
                    -(d3 @ r_xi_dot) / ji,
                    (d2 @ r_xi_dot) / ji,
                ]
            )

            return A, B, K_Omega

        def Omega_C(e1, d1, phi, phi_dot, r_xi_dot, ji):
            e_complement = cross3(-e1, d1) / norm(cross3(-e1, d1))
            A_pi = rodriguez(e_complement * np.pi)
            A = smallest_rotation(-e1, d1) @ A_pi
            B = rodriguez(d1 * phi)
            R = B @ A
            d1, d2, d3 = R.T

            K_Omega = np.array(
                [
                    phi_dot
                    - r_xi_dot
                    @ cross3(d1, -e1)
                    / (ji * (1 + d1 @ e1)),  # Mitterbach2020 (2.105)
                    -(d3 @ r_xi_dot) / ji,
                    (d2 @ r_xi_dot) / ji,
                ]
            )

            return A, B, K_Omega

        if switching_beam:
            # check sign of inner product
            cos_theta = e1 @ d1
            sign_cos = sign(cos_theta)
        else:
            sign_cos = 1.0

        if sign_cos >= 0:
            A, B, K_Omega = Omega(e1, d1, phi, phi_dot, r_xi_dot, ji)
        else:
            # A, B, K_Omega = Omega(e1, d1, phi, phi_dot, r_xi_dot, ji)
            # print(f"A:\n{A}")
            # print(f"B:\n{B}")
            # print(f"K_Omega:\n{K_Omega}")
            A, B, K_Omega = Omega_C(e1, d1, phi, phi_dot, r_xi_dot, ji)
            # print(f"A:\n{A}")
            # print(f"B:\n{B}")
            # print(f"K_Omega:\n{K_Omega}")

            # print(f"")

        return K_Omega

        # # build rotation matrices
        # A = smallest_rotation(e1, d1)
        # B = rodriguez(d1 * phi)
        # R = B @ A
        # d1, d2, d3 = R.T

        # return np.array(
        #     [
        #         phi_dot
        #         + r_xi_dot
        #         @ cross3(d1, e1)
        #         / (ji * (1 + d1 @ e1)),  # Mitterbach2020 (2.105)
        #         -(d3 @ r_xi_dot) / ji,
        #         (d2 @ r_xi_dot) / ji,
        #     ]
        # )

    # TODO:
    def K_J_R(self, t, q, frame_ID):
        return approx_fprime(
            np.zeros_like(q),
            lambda u: self.K_Omega(t, q, u, frame_ID),
            method="3-point",
        )

    # TODO:
    def K_J_R_q(self, t, q, frame_ID):
        return approx_fprime(q, lambda q: self.K_J_R(t, q, frame_ID), method="2-point")

    # TODO:
    def K_Psi(self, t, q, u, u_dot, frame_ID):
        raise NotImplementedError("")

    ############
    # body force
    ############
    def distributed_force1D_el(self, force, t, el):
        fe = np.zeros(self.nq_el)
        for i in range(self.nQP):
            NNi = self.stack3r(self.N_r[el, i])
            fe[self.rDOF] += (
                NNi.T @ force(t, self.qp[el, i]) * self.J0[el, i] * self.qw[el, i]
            )
        return fe

    def distributed_force1D(self, t, q, force):
        f = np.zeros(self.nq)
        for el in range(self.nEl):
            f[self.elDOF[el]] += self.distributed_force1D_el(force, t, el)
        return f

    # def body_force_q(self, t, q, coo, force):
    def distributed_force1D_q(self, t, q, coo, force):
        pass

    ####################################################
    # visualization
    ####################################################
    def nodes(self, q):
        return q[self.qDOF][: 3 * self.nn_r].reshape(3, -1)

    def centerline(self, q, n=100):
        q_body = q[self.qDOF]
        r = []
        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))
        return np.array(r).T

    def frames(self, q, n=10):
        q_body = q[self.qDOF]
        r = []
        d1 = []
        d2 = []
        d3 = []

        for xi in np.linspace(0, 1, n):
            frame_ID = (xi,)
            qp = q_body[self.qDOF_P(frame_ID)]
            r.append(self.r_OP(1, qp, frame_ID))

            d1i, d2i, d3i = self.A_IK(1, qp, frame_ID).T
            d1.extend([d1i])
            d2.extend([d2i])
            d3.extend([d3i])

        return np.array(r).T, np.array(d1).T, np.array(d2).T, np.array(d3).T

    # def plot_centerline(self, ax, q, n=100, color="black"):
    #     ax.plot(*self.nodes(q), linestyle="dashed", marker="o", color=color)
    #     ax.plot(*self.centerline(q, n=n), linestyle="solid", color=color)

    # def plot_frames(self, ax, q, n=10, length=1):
    #     r, d1, d2, d3 = self.frames(q, n=n)
    #     ax.quiver(*r, *d1, color="red", length=length)
    #     ax.quiver(*r, *d2, color="green", length=length)
    #     ax.quiver(*r, *d3, color="blue", length=length)

    # ############
    # # vtk export
    # ############
    # def post_processing_vtk_volume_circle(self, t, q, filename, R, binary=False):
    #     # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
    #     assert (
    #         self.polynomial_degree_r == self.polynomial_degree_phi
    #     ), "Not implemented for mixed polynomial degrees"

    #     # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    #     nn_xi = self.nEl + self.polynomial_degree_r
    #     nEl_eta = 1
    #     nEl_zeta = 4
    #     # see Cotrell2009 Section 2.4.2
    #     # TODO: Maybe eta and zeta have to be interchanged
    #     polynomial_degree_eta = 1
    #     polynomial_degree_zeta = 2
    #     nn_eta = nEl_eta + polynomial_degree_eta
    #     nn_zeta = nEl_zeta + polynomial_degree_zeta

    #     # # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
    #     # as_ = np.linspace(-a/2, a/2, num=nn_eta, endpoint=True)
    #     # bs_ = np.linspace(-b/2, b/2, num=nn_eta, endpoint=True)

    #     circle_points = (
    #         0.5
    #         * R
    #         * np.array(
    #             [
    #                 [1, 0, 0],
    #                 [1, 1, 0],
    #                 [0, 1, 0],
    #                 [-1, 1, 0],
    #                 [-1, 0, 0],
    #                 [-1, -1, 0],
    #                 [0, -1, 0],
    #                 [1, -1, 0],
    #                 [1, 0, 0],
    #             ],
    #             dtype=float,
    #         )
    #     )

    #     Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
    #     for i in range(self.nn_r):
    #         qr = q[self.nodalDOF_r[i]]
    #         q_di = q[self.nodalDOF_phi[i]]
    #         A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

    #         for k, point in enumerate(circle_points):
    #             # Note: eta index is always 0!
    #             Pw[i, 0, k] = qr + A_IK @ point

    #     if self.basis == "B-spline":
    #         knot_vector_eta = Knot_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Knot_vector(polynomial_degree_zeta, nEl_zeta)
    #     elif self.basis == "lagrange":
    #         knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
    #     knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
    #     degrees = (
    #         self.polynomial_degree_r,
    #         polynomial_degree_eta,
    #         polynomial_degree_zeta,
    #     )

    #     # Build Bezier patches from B-spline control points
    #     from cardillo.discretization.B_spline import decompose_B_spline_volume

    #     Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

    #     nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

    #     # build vtk mesh
    #     n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
    #     patch_size = p1 * q1 * r1
    #     points = np.zeros((n_patches * patch_size, dim))
    #     cells = []
    #     HigherOrderDegrees = []
    #     RationalWeights = []
    #     vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
    #     from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

    #     for i in range(nbezier_xi):
    #         for j in range(nbezier_eta):
    #             for k in range(nbezier_zeta):
    #                 idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
    #                 point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
    #                 points[point_range] = rearange_vtk3D(Qw[i, j, k])

    #                 cells.append((vtk_cell_type, point_range[None]))
    #                 HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
    #                 weight = np.sqrt(2) / 2
    #                 # tmp = np.array([np.sqrt(2) / 2, 1.0])
    #                 # RationalWeights.append(np.tile(tmp, 8)[None])
    #                 weights_vertices = weight * np.ones(8)
    #                 weights_edges = np.ones(4 * nn_xi)
    #                 weights_faces = np.ones(2)
    #                 weights_volume = np.ones(nn_xi - 2)
    #                 weights = np.concatenate(
    #                     (weights_edges, weights_vertices, weights_faces, weights_volume)
    #                 )
    #                 # weights = np.array([weight, weight, weight, weight,
    #                 #                     1.0,    1.0,    1.0,    1.0,
    #                 #                     0.0,    0.0,    0.0,    0.0 ])
    #                 weights = np.ones_like(point_range)
    #                 RationalWeights.append(weights[None])

    #     # RationalWeights = np.ones(len(points))
    #     RationalWeights = 2 * (np.random.rand(len(points)) + 1)

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         # filename.parent / (filename.stem + '.vtu'),
    #         filename,
    #         points,
    #         cells,
    #         point_data={
    #             "RationalWeights": RationalWeights,
    #         },
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees},
    #         binary=binary,
    #     )

    # def post_processing_vtk_volume(self, t, q, filename, circular=True, binary=False):
    #     # This is mandatory, otherwise we cannot construct the 3D continuum without L2 projection!
    #     assert (
    #         self.polynomial_degree_r == self.polynomial_degree_phi
    #     ), "Not implemented for mixed polynomial degrees"

    #     # rearrange generalized coordinates from solver ordering to Piegl's Pw 3D array
    #     nn_xi = self.nEl + self.polynomial_degree_r
    #     nEl_eta = 1
    #     nEl_zeta = 1
    #     if circular:
    #         polynomial_degree_eta = 2
    #         polynomial_degree_zeta = 2
    #     else:
    #         polynomial_degree_eta = 1
    #         polynomial_degree_zeta = 1
    #     nn_eta = nEl_eta + polynomial_degree_eta
    #     nn_zeta = nEl_zeta + polynomial_degree_zeta

    #     # TODO: We do the hard coded case for rectangular cross section here, but this has to be extended to the circular cross section case too!
    #     if circular:
    #         r = 0.2
    #         a = b = r
    #     else:
    #         a = 0.2
    #         b = 0.1
    #     as_ = np.linspace(-a / 2, a / 2, num=nn_eta, endpoint=True)
    #     bs_ = np.linspace(-b / 2, b / 2, num=nn_eta, endpoint=True)

    #     Pw = np.zeros((nn_xi, nn_eta, nn_zeta, 3))
    #     for i in range(self.nn_r):
    #         qr = q[self.nodalDOF_r[i]]
    #         q_di = q[self.nodalDOF_phi[i]]
    #         A_IK = q_di.reshape(3, 3, order="F")  # TODO: Check this!

    #         for j, aj in enumerate(as_):
    #             for k, bk in enumerate(bs_):
    #                 Pw[i, j, k] = qr + A_IK @ np.array([0, aj, bk])

    #     if self.basis == "B-spline":
    #         knot_vector_eta = Knot_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Knot_vector(polynomial_degree_zeta, nEl_zeta)
    #     elif self.basis == "lagrange":
    #         knot_vector_eta = Node_vector(polynomial_degree_eta, nEl_eta)
    #         knot_vector_zeta = Node_vector(polynomial_degree_zeta, nEl_zeta)
    #     knot_vector_objs = [self.knot_vector_r, knot_vector_eta, knot_vector_zeta]
    #     degrees = (
    #         self.polynomial_degree_r,
    #         polynomial_degree_eta,
    #         polynomial_degree_zeta,
    #     )

    #     # Build Bezier patches from B-spline control points
    #     from cardillo.discretization.B_spline import decompose_B_spline_volume

    #     Qw = decompose_B_spline_volume(knot_vector_objs, Pw)

    #     nbezier_xi, nbezier_eta, nbezier_zeta, p1, q1, r1, dim = Qw.shape

    #     # build vtk mesh
    #     n_patches = nbezier_xi * nbezier_eta * nbezier_zeta
    #     patch_size = p1 * q1 * r1
    #     points = np.zeros((n_patches * patch_size, dim))
    #     cells = []
    #     HigherOrderDegrees = []
    #     RationalWeights = []
    #     vtk_cell_type = "VTK_BEZIER_HEXAHEDRON"
    #     from PyPanto.miscellaneous.indexing import flat3D, rearange_vtk3D

    #     for i in range(nbezier_xi):
    #         for j in range(nbezier_eta):
    #             for k in range(nbezier_zeta):
    #                 idx = flat3D(i, j, k, (nbezier_xi, nbezier_eta))
    #                 point_range = np.arange(idx * patch_size, (idx + 1) * patch_size)
    #                 points[point_range] = rearange_vtk3D(Qw[i, j, k])

    #                 cells.append((vtk_cell_type, point_range[None]))
    #                 HigherOrderDegrees.append(np.array(degrees, dtype=float)[None])
    #                 weight = np.sqrt(2) / 2
    #                 # tmp = np.array([np.sqrt(2) / 2, 1.0])
    #                 # RationalWeights.append(np.tile(tmp, 8)[None])
    #                 weights_vertices = weight * np.ones(8)
    #                 weights_edges = np.ones(4 * nn_xi)
    #                 weights_faces = np.ones(2)
    #                 weights_volume = np.ones(nn_xi - 2)
    #                 weights = np.concatenate(
    #                     (weights_edges, weights_vertices, weights_faces, weights_volume)
    #                 )
    #                 # weights = np.array([weight, weight, weight, weight,
    #                 #                     1.0,    1.0,    1.0,    1.0,
    #                 #                     0.0,    0.0,    0.0,    0.0 ])
    #                 weights = np.ones_like(point_range)
    #                 RationalWeights.append(weights[None])

    #     # RationalWeights = np.ones(len(points))
    #     RationalWeights = 2 * (np.random.rand(len(points)) + 1)

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         # filename.parent / (filename.stem + '.vtu'),
    #         filename,
    #         points,
    #         cells,
    #         point_data={
    #             "RationalWeights": RationalWeights,
    #         },
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees},
    #         binary=binary,
    #     )

    # def post_processing(self, t, q, filename, binary=True):
    #     # write paraview PVD file collecting time and all vtk files, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    #     from xml.dom import minidom

    #     root = minidom.Document()

    #     vkt_file = root.createElement("VTKFile")
    #     vkt_file.setAttribute("type", "Collection")
    #     root.appendChild(vkt_file)

    #     collection = root.createElement("Collection")
    #     vkt_file.appendChild(collection)

    #     for i, (ti, qi) in enumerate(zip(t, q)):
    #         filei = filename + f"{i}.vtu"

    #         # write time step and file name in pvd file
    #         dataset = root.createElement("DataSet")
    #         dataset.setAttribute("timestep", f"{ti:0.6f}")
    #         dataset.setAttribute("file", filei)
    #         collection.appendChild(dataset)

    #         self.post_processing_single_configuration(ti, qi, filei, binary=binary)

    #     # write pvd file
    #     xml_str = root.toprettyxml(indent="\t")
    #     with open(filename + ".pvd", "w") as f:
    #         f.write(xml_str)

    # def post_processing_single_configuration(self, t, q, filename, binary=True):
    #     # centerline and connectivity
    #     cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

    #     # if the centerline and the directors are interpolated with the same
    #     # polynomial degree we can use the values on the nodes and decompose the B-spline
    #     # into multiple Bezier patches, otherwise the directors have to be interpolated
    #     # onto the nodes of the centerline by a so-called L2-projection, see below
    #     same_shape_functions = False
    #     if self.polynomial_degree_r == self.polynomial_degree_phi:
    #         same_shape_functions = True

    #     if same_shape_functions:
    #         _, points_di, _ = self.mesh_phi.vtk_mesh(q[self.nq_r :])

    #         # fill dictionary storing point data with directors
    #         point_data = {
    #             "d1": points_di[:, 0:3],
    #             "d2": points_di[:, 3:6],
    #             "d3": points_di[:, 6:9],
    #         }

    #     else:
    #         point_data = {}

    #     # export existing values on quadrature points using L2 projection
    #     J0_vtk = self.mesh_r.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
    #     point_data.update({"J0": J0_vtk})

    #     Gamma0_vtk = self.mesh_r.field_to_vtk(self.lambda0)
    #     point_data.update({"Gamma0": Gamma0_vtk})

    #     Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
    #     point_data.update({"Kappa0": Kappa0_vtk})

    #     # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
    #     # - strain measures Gamma & Kappa
    #     # - directors d1, d2, d3
    #     Gamma = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     Kappa = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     if not same_shape_functions:
    #         d1s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d2s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d3s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     for el in range(self.nEl):
    #         qe = q[self.elDOF[el]]

    #         # extract generalized coordinates for beam centerline and directors
    #         # in the current and reference configuration
    #         qe_r = qe[self.rDOF]
    #         qe_d1 = qe[self.phiDOF]
    #         qe_d2 = qe[self.d2DOF]
    #         qe_d3 = qe[self.d3DOF]

    #         for i in range(self.nQP):
    #             # build matrix of shape function derivatives
    #             NN_di_i = self.stack3di(self.N_phi[el, i])
    #             NN_r_xii = self.stack3r(self.N_r_xi[el, i])
    #             NN_di_xii = self.stack3di(self.N_phi_xi[el, i])

    #             # extract reference state variables
    #             J0i = self.J0[el, i]
    #             Gamma0_i = self.lambda0[el, i]
    #             Kappa0_i = self.Kappa0[el, i]

    #             # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
    #             # the parameter space \xi and thus need to be transformed later
    #             r_xi = NN_r_xii @ qe_r

    #             d1 = NN_di_i @ qe_d1
    #             d1_xi = NN_di_xii @ qe_d1

    #             d2 = NN_di_i @ qe_d2
    #             d2_xi = NN_di_xii @ qe_d2

    #             d3 = NN_di_i @ qe_d3
    #             d3_xi = NN_di_xii @ qe_d3

    #             # compute derivatives w.r.t. the arc lenght parameter s
    #             r_s = r_xi / J0i

    #             d1_s = d1_xi / J0i
    #             d2_s = d2_xi / J0i
    #             d3_s = d3_xi / J0i

    #             # build rotation matrices
    #             if not same_shape_functions:
    #                 d1s[el, i] = d1
    #                 d2s[el, i] = d2
    #                 d3s[el, i] = d3
    #             R = np.vstack((d1, d2, d3)).T

    #             # axial and shear strains
    #             Gamma[el, i] = R.T @ r_s

    #             # torsional and flexural strains
    #             Kappa[el, i] = np.array(
    #                 [
    #                     0.5 * (d3 @ d2_s - d2 @ d3_s),
    #                     0.5 * (d1 @ d3_s - d3 @ d1_s),
    #                     0.5 * (d2 @ d1_s - d1 @ d2_s),
    #                 ]
    #             )

    #     # L2 projection of strain measures
    #     Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
    #     point_data.update({"Gamma": Gamma_vtk})

    #     Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
    #     point_data.update({"Kappa": Kappa_vtk})

    #     # L2 projection of directors
    #     if not same_shape_functions:
    #         d1_vtk = self.mesh_r.field_to_vtk(d1s)
    #         point_data.update({"d1": d1_vtk})
    #         d2_vtk = self.mesh_r.field_to_vtk(d2s)
    #         point_data.update({"d2": d2_vtk})
    #         d3_vtk = self.mesh_r.field_to_vtk(d3s)
    #         point_data.update({"d3": d3_vtk})

    #     # fields depending on strain measures and other previously computed quantities
    #     point_data_fields = {
    #         "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
    #             [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
    #         ),
    #         "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #         "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #     }

    #     for name, fun in point_data_fields.items():
    #         tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
    #             -1
    #         )
    #         field = np.zeros((len(Gamma_vtk), len(tmp)))
    #         for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
    #             zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
    #         ):
    #             field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
    #         point_data.update({name: field})

    #     # write vtk mesh using meshio
    #     meshio.write_points_cells(
    #         os.path.splitext(os.path.basename(filename))[0] + ".vtu",
    #         points_r,  # only export centerline as geometry here!
    #         cells_r,
    #         point_data=point_data,
    #         cell_data={"HigherOrderDegrees": HigherOrderDegrees_r},
    #         binary=binary,
    #     )

    # def post_processing_subsystem(self, t, q, u, binary=True):
    #     # centerline and connectivity
    #     cells_r, points_r, HigherOrderDegrees_r = self.mesh_r.vtk_mesh(q[: self.nq_r])

    #     # if the centerline and the directors are interpolated with the same
    #     # polynomial degree we can use the values on the nodes and decompose the B-spline
    #     # into multiple Bezier patches, otherwise the directors have to be interpolated
    #     # onto the nodes of the centerline by a so-called L2-projection, see below
    #     same_shape_functions = False
    #     if self.polynomial_degree_r == self.polynomial_degree_phi:
    #         same_shape_functions = True

    #     if same_shape_functions:
    #         _, points_di, _ = self.mesh_phi.vtk_mesh(q[self.nq_r :])

    #         # fill dictionary storing point data with directors
    #         point_data = {
    #             "u_r": points_r - points_r[0],
    #             "d1": points_di[:, 0:3],
    #             "d2": points_di[:, 3:6],
    #             "d3": points_di[:, 6:9],
    #         }

    #     else:
    #         point_data = {}

    #     # export existing values on quadrature points using L2 projection
    #     J0_vtk = self.mesh_r.field_to_vtk(self.J0.reshape(self.nEl, self.nQP, 1))
    #     point_data.update({"J0": J0_vtk})

    #     Gamma0_vtk = self.mesh_r.field_to_vtk(self.lambda0)
    #     point_data.update({"Gamma0": Gamma0_vtk})

    #     Kappa0_vtk = self.mesh_r.field_to_vtk(self.Kappa0)
    #     point_data.update({"Kappa0": Kappa0_vtk})

    #     # evaluate fields at quadrature points that have to be projected onto the centerline mesh:
    #     # - strain measures Gamma & Kappa
    #     # - directors d1, d2, d3
    #     Gamma = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     Kappa = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     if not same_shape_functions:
    #         d1s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d2s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #         d3s = np.zeros((self.mesh_r.nel, self.mesh_r.nqp, 3))
    #     for el in range(self.nEl):
    #         qe = q[self.elDOF[el]]

    #         # extract generalized coordinates for beam centerline and directors
    #         # in the current and reference configuration
    #         qe_r = qe[self.rDOF]
    #         qe_d1 = qe[self.phiDOF]
    #         qe_d2 = qe[self.d2DOF]
    #         qe_d3 = qe[self.d3DOF]

    #         for i in range(self.nQP):
    #             # build matrix of shape function derivatives
    #             NN_di_i = self.stack3di(self.N_phi[el, i])
    #             NN_r_xii = self.stack3r(self.N_r_xi[el, i])
    #             NN_di_xii = self.stack3di(self.N_phi_xi[el, i])

    #             # extract reference state variables
    #             J0i = self.J0[el, i]
    #             Gamma0_i = self.lambda0[el, i]
    #             Kappa0_i = self.Kappa0[el, i]

    #             # Interpolate necessary quantities. The derivatives are evaluated w.r.t.
    #             # the parameter space \xi and thus need to be transformed later
    #             r_xi = NN_r_xii @ qe_r

    #             d1 = NN_di_i @ qe_d1
    #             d1_xi = NN_di_xii @ qe_d1

    #             d2 = NN_di_i @ qe_d2
    #             d2_xi = NN_di_xii @ qe_d2

    #             d3 = NN_di_i @ qe_d3
    #             d3_xi = NN_di_xii @ qe_d3

    #             # compute derivatives w.r.t. the arc lenght parameter s
    #             r_s = r_xi / J0i

    #             d1_s = d1_xi / J0i
    #             d2_s = d2_xi / J0i
    #             d3_s = d3_xi / J0i

    #             # build rotation matrices
    #             if not same_shape_functions:
    #                 d1s[el, i] = d1
    #                 d2s[el, i] = d2
    #                 d3s[el, i] = d3
    #             R = np.vstack((d1, d2, d3)).T

    #             # axial and shear strains
    #             Gamma[el, i] = R.T @ r_s

    #             # torsional and flexural strains
    #             Kappa[el, i] = np.array(
    #                 [
    #                     0.5 * (d3 @ d2_s - d2 @ d3_s),
    #                     0.5 * (d1 @ d3_s - d3 @ d1_s),
    #                     0.5 * (d2 @ d1_s - d1 @ d2_s),
    #                 ]
    #             )

    #     # L2 projection of strain measures
    #     Gamma_vtk = self.mesh_r.field_to_vtk(Gamma)
    #     point_data.update({"Gamma": Gamma_vtk})

    #     Kappa_vtk = self.mesh_r.field_to_vtk(Kappa)
    #     point_data.update({"Kappa": Kappa_vtk})

    #     # L2 projection of directors
    #     if not same_shape_functions:
    #         d1_vtk = self.mesh_r.field_to_vtk(d1s)
    #         point_data.update({"d1": d1_vtk})
    #         d2_vtk = self.mesh_r.field_to_vtk(d2s)
    #         point_data.update({"d2": d2_vtk})
    #         d3_vtk = self.mesh_r.field_to_vtk(d3s)
    #         point_data.update({"d3": d3_vtk})

    #     # fields depending on strain measures and other previously computed quantities
    #     point_data_fields = {
    #         "W": lambda Gamma, Gamma0, Kappa, Kappa0: np.array(
    #             [self.material_model.potential(Gamma, Gamma0, Kappa, Kappa0)]
    #         ),
    #         "n_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.n_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #         "m_i": lambda Gamma, Gamma0, Kappa, Kappa0: self.material_model.m_i(
    #             Gamma, Gamma0, Kappa, Kappa0
    #         ),
    #     }

    #     for name, fun in point_data_fields.items():
    #         tmp = fun(Gamma_vtk[0], Gamma0_vtk[0], Kappa_vtk[0], Kappa0_vtk[0]).reshape(
    #             -1
    #         )
    #         field = np.zeros((len(Gamma_vtk), len(tmp)))
    #         for i, (Gamma_i, Gamma0_i, Kappa_i, Kappa0_i) in enumerate(
    #             zip(Gamma_vtk, Gamma0_vtk, Kappa_vtk, Kappa0_vtk)
    #         ):
    #             field[i] = fun(Gamma_i, Gamma0_i, Kappa_i, Kappa0_i).reshape(-1)
    #         point_data.update({name: field})

    #     return points_r, point_data, cells_r, HigherOrderDegrees_r
