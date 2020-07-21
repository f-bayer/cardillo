import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, bmat
from tqdm import tqdm

from cardillo.math.numerical_derivative import Numerical_derivative
from cardillo.solver import Solution
from cardillo.math.prox import prox_Rn0

class Moreau():
    def __init__(self, model, t1, dt):
        self.model = model

        self.fix_point_error_function = lambda x: np.max(np.abs(x)) / len(x)
        self.fix_point_tol = 1e-3
        self.fix_point_max_iter = 1000


        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.n = self.nq + self.nu + self.nla_g + self.nla_gamma

    def step(self, tk, qk, uk, la_Nk, la_Tk):
        # general quantities
        dt = self.dt

        tk1 = tk + dt
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)

        M = self.model.M(tk1, qk1)
        h = self.model.h(tk1, qk1, uk)
        W_g = self.model.W_g(tk1, qk1)
        W_gamma = self.model.W_gamma(tk1, qk1)
        W_N = self.model.W_N(tk1, qk1, scipy_matrix=csc_matrix)
        W_T = self.model.W_T(tk1, qk1, scipy_matrix=csc_matrix)
        g_dot_u = self.model.g_dot_u(tk1, qk1)
        chi_g = self.model.chi_g(tk1, qk1)
        gamma_u = self.model.gamma_u(tk1, qk1)
        chi_gamma = self.model.chi_gamma(tk1, qk1)
        
        # M (uk1 - uk) - dt (h + W_g la_g + W_gamma la_gamma + W_gN la_N + W_gT la_T) = 0
        # g_dot_u @ uk1 + chi_g = 0
        # gamma_u @ uk1 + chi_gamma = 0
        g_N = self.model.g_N(tk1, qk1)
        I_N = (g_N <= 0)
        I_T = self.model.NT_connectivity[I_N].reshape(-1)
        # solve for new velocities and bilateral constraint forces
        A =  bmat([[M      ,  -dt * W_g, -dt * W_gamma], \
                    [g_dot_u,       None,          None], \
                    [gamma_u,       None,          None]]).tocsr()

        b = np.concatenate( (M @ uk + dt * h + W_N[:, I_N] @ la_Nk[I_N] + W_T[:, I_T] @ la_Tk[I_T],\
                                -chi_g,\
                                -chi_gamma) )

        x = spsolve(A, b)
        uk1 = x[:self.nu]
        la_gk1 = x[self.nu:self.nu+self.nla_g]
        la_gammak1 = x[self.nu+self.nla_g:]

        la_Nk1 = np.zeros_like(la_Nk)
        la_Tk1 = np.zeros_like(la_Tk)

        converged = True
        error = 0
        j = 0
        if np.any(I_N):
            #TODO:
            # I_T = self.model.la_TDOF
            converged = False
            la_Nk0 = la_Nk.copy()
            la_Nk1_i = la_Nk.copy()
            la_Tk0 = la_Tk.copy()
            la_Tk1_i = la_Tk.copy()
            for j in range(self.fix_point_max_iter):
                
                # relative contact velocity and prox equation
                la_Nk1_i, la_Tk1_i = self.model.contact_force_fixpoint_update(tk1, qk1, uk, uk1, la_Nk1_i, la_Tk1_i, I_N=I_N)

                # check if velocities or contact percussions do not change
                # error = self.fix_point_error_function(uk1 - uk0)
                R = np.concatenate( (la_Nk1_i[I_N] - la_Nk0[I_N], la_Tk1_i[I_T] - la_Tk0[I_T]) )
                error = self.fix_point_error_function(R)
                converged = error < self.fix_point_tol
                if converged:
                    la_Nk1[I_N] = la_Nk1_i[I_N]
                    la_Tk1[I_T] = la_Tk1_i[I_T]
                    break
                la_Nk0 = la_Nk1_i.copy()
                la_Tk0 = la_Tk1_i.copy()

                # solve for new velocities and bilateral constraint forces
                A =  bmat([[M      ,  -dt * W_g, -dt * W_gamma], \
                        [g_dot_u,       None,          None], \
                        [gamma_u,       None,          None]]).tocsr()

                b = np.concatenate( (M @ uk + dt * h + W_N[:, I_N] @ la_Nk1_i[I_N] + W_T[:, I_T] @ la_Tk1_i[I_T],\
                                    -chi_g,\
                                    -chi_gamma) )

                x = spsolve(A, b)
                uk1 = x[:self.nu]
                la_gk1 = x[self.nu:self.nu+self.nla_g]
                la_gammak1 = x[self.nu+self.nla_g:]
                
        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1

    def solve(self): 
        
        # lists storing output variables
        tk = self.model.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        la_gk = self.model.la_g0.copy()
        la_gammak = self.model.la_gamma0.copy()
        la_Nk = self.model.la_N0.copy()
        la_Tk = self.model.la_T0.copy()
        
        q = [qk]
        u = [uk]
        la_g = [la_gk]
        la_gamma = [la_gammak]
        la_N = [la_Nk]
        la_T = [la_Tk]

        pbar = tqdm(self.t[:-1])
        for tk in pbar:
            (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1 = self.step(tk, qk, uk, la_Nk, la_Tk)
            pbar.set_description(f't: {tk1:0.2e}; fixed point iterations: {j+1}; error: {error:.3e}')
            if not converged:
                raise RuntimeError(f'internal fixed point iteration not converged after {j+1} iterations with error: {error:.5e}')

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)
            la_N.append(la_Nk1)
            la_T.append(la_Tk1)

            # update local variables for accepted time step
            qk, uk, la_gk, la_gammak, la_Nk, la_Tk = qk1, uk1, la_gk1, la_gammak1, la_Nk1, la_Tk1
            
        # write solution
        #TODO:
        return Solution(t=self.t, q=np.array(q), u=np.array(u), la_g=np.array(la_g), la_gamma=np.array(la_gamma), la_N=np.array(la_N))

class Moreau_sym():
    def __init__(self, model, t1, dt, newton_tol=1e-6, newton_max_iter=10, newton_error_function=lambda x: np.max(np.abs(x)), numerical_jacobian=False, debug=False):
        
        self.model = model

        # integration time
        t0 = model.t0
        self.t1 = t1 if t1 > t0 else ValueError("t1 must be larger than initial time t0.")
        self.dt = dt
        self.t = np.arange(t0, self.t1 + self.dt, self.dt)

        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.newton_error_function = newton_error_function
        self.linearSolver = spsolve

        self.nq = self.model.nq
        self.nu = self.model.nu
        self.nla_g = self.model.nla_g
        self.nla_gamma = self.model.nla_gamma
        self.n = self.nu + self.nla_g + self.nla_gamma

        self.uDOF = np.arange(self.nu)
        self.la_gDOF = self.nu + np.arange(self.nla_g)
        self.la_gammaDOF = self.nu + self.nla_g + np.arange(self.nla_gamma)

        self.Mk1 = self.model.M(t0, model.q0)
        self.W_gk1 = self.model.W_g(t0, model.q0)
        self.W_gammak1 = self.model.W_gamma(t0, model.q0)

        self.numerical_jacobian = numerical_jacobian
        if numerical_jacobian:
            self.__R_x = self.__R_x_num
        else:
            self.__R_x = self.__R_x_analytic
        self.debug = debug
        if debug:
            self.__R_x = self.__R_x_debug

    def __R(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        self.Mk1 = self.model.M(tk1, qk1)
        self.W_gk1 = self.model.W_g(tk1, qk1)
        self.W_gammak1 = self.model.W_gamma(tk1, qk1)

        R = np.zeros(self.n)
        R[self.uDOF] = self.Mk1 @ (uk1 - uk) - 0.5 * self.dt * (self.model.h(tk1, qk1, uk) + self.model.h(tk1, qk1, uk1) ) - self.dt * ( self.W_gk1 @ la_gk1 + self.W_gammak1 @ la_gammak1 )
        R[self.la_gDOF] = self.model.g_dot(tk1, qk1, uk1)
        R[self.la_gammaDOF] = self.model.gamma(tk1, qk1, uk1)

        return R

    def __R_wrapper(self, t, x, y):
        
        uk1 = x[self.uDOF]
        la_gk1 = x[self.la_gDOF]
        la_gammak1 = x[self.la_gammaDOF]

        qk1 = y[:self.nq]
        uk = y[self.nq:]

        return self.__R(uk, t, qk1, uk1, la_gk1, la_gammak1)

    def __R_x_num(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        y = np.zeros(self.nq + self.nu)
        y[:self.nq] = qk1
        y[self.nq:] = uk

        x = np.zeros(self.n)
        x[self.uDOF] = uk1
        x[self.la_gDOF] = la_gk1
        x[self.la_gammaDOF] = la_gammak1

        R_x_num = Numerical_derivative(self.__R_wrapper, order=2)._x(tk1, x, y)

        return csr_matrix( R_x_num )

    def __R_x_analytic(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        # equations of motion
        Ru_u = self.Mk1 - 0.5* self.dt * self.model.h_u(tk1, qk1, uk1)
        Ru_la_g = - self.dt * self.W_gk1
        Ru_la_gamma = - self.dt * self.W_gammak1

        # constraint equations
        Rla_g_u = self.model.g_dot_u(tk1, qk1)
        Rla_gamma_u = self.model.gamma_u(tk1, qk1)
        
        return bmat([[Ru_u       , Ru_la_g, Ru_la_gamma], \
                     [Rla_g_u    ,    None,        None], \
                     [Rla_gamma_u,    None,        None]]).tocsc()

    def __R_x_debug(self, uk, tk1, qk1, uk1, la_gk1, la_gammak1):
        R_x_num = self.__R_x_num(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        R_x_analytic = self.__R_x_analytic(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        diff = R_x_num - R_x_analytic.toarray()

        print(f'total error jacobian: {np.linalg.norm(diff)/ self.n:.5e}')

        if self.numerical_jacobian:
            return R_x_num
        else:
            return R_x_analytic
        
    def step(self, tk, qk, uk, la_gk, la_gammak):
        dt = self.dt
        tk1 = tk + dt

        # foward Euler predictor
        la_gk1 = la_gk
        la_gammak1 = la_gammak
        uk1 = uk + dt * spsolve(self.Mk1.tocsc(), self.model.h(tk, qk, uk) + self.W_gk1 @ la_gk + self.W_gammak1 @ la_gammak)
        qk1 = qk + dt * self.model.q_dot(tk, qk, uk)

        # initial guess for Newton-Raphson solver
        xk1 = np.zeros(self.n)
        xk1[self.uDOF] = uk1
        xk1[self.la_gDOF] = la_gk1
        xk1[self.la_gammaDOF] = la_gammak1

        # initial residual and error
        R = self.__R(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
        error = self.newton_error_function(R)
        converged = error < self.newton_tol
        j = 0
        if not converged:
            while j < self.newton_max_iter:
                # jacobian
                R_x = self.__R_x(uk, tk1, qk1, uk1, la_gk1, la_gammak1)

                # Newton update
                j += 1
                dx = spsolve(R_x, R)
                xk1 -= dx
                uk1 = xk1[self.uDOF]
                la_gk1 = xk1[self.la_gDOF]
                la_gammak1 = xk1[self.la_gammaDOF]

                R = self.__R(uk, tk1, qk1, uk1, la_gk1, la_gammak1)
                error = self.newton_error_function(R)
                converged = error < self.newton_tol
                if converged:
                    break

            if not converged:
                raise RuntimeError(f'internal Newton-Raphson method not converged after {j} stepts with error: {error:.5e}')
            
        return (converged, j, error), tk1, qk1, uk1, la_gk1, la_gammak1

    def solve(self): 
        # lists storing output variables
        tk = self.model.t0
        qk = self.model.q0.copy()
        uk = self.model.u0.copy()
        la_gk = self.model.la_g0.copy()
        la_gammak = self.model.la_gamma0.copy()
        
        q = [qk]
        u = [uk]
        la_g = [la_gk]
        la_gamma = [la_gammak]


        pbar = tqdm(self.t[:-1])
        for tk in pbar:
            (converged, n_iter, error), tk1, qk1, uk1, la_gk1, la_gammak1 = self.step(tk, qk, uk, la_gk, la_gammak)

            pbar.set_description(f't: {tk1:0.2e}; Newton: {n_iter}/{self.newton_max_iter} iterations; error: {error:0.2e}')

            qk1, uk1 = self.model.solver_step_callback(tk1, qk1, uk1)

            q.append(qk1)
            u.append(uk1)
            la_g.append(la_gk1)
            la_gamma.append(la_gammak1)

            # update local variables for accepted time step
            qk, uk, la_gk, la_gammak = qk1, uk1, la_gk1, la_gammak1
            
        # write solution
        return Solution(t=self.t, q=np.array(q), u=np.array(u), la_g=np.array(la_g), la_gamma=np.array(la_gamma))