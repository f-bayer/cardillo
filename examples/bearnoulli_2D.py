from cardillo.model.classical_beams import Hooke, Euler_bernoulli2D
from cardillo.model.frame import Frame
from cardillo.model.bilateral_constraints import Spherical_joint
from cardillo.model import Model
from cardillo.solver import Euler_backward, Moreau, Moreau_sym, Generalized_alpha_1, Scipy_ivp, Newton
from cardillo.model.line_force.line_force import Line_force
from cardillo.discretization import uniform_knot_vector

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import numpy as np

statics = True
# statics = False

if __name__ == "__main__":
    # physical properties of the rope
    L = 50
    r = 3.0e-3
    A = np.pi * r**2
    EA = 4.0e8 * A * 1.0e-2
    EI = 1.0e4
    material_model = Hooke(EA, EI)
    A_rho0 = 10 * A

    # discretization properties
    p = 2
    assert p >= 2
    nQP = int(np.ceil((p + 1)**2 / 2))
    print(f'nQP: {nQP}')
    nEl = 10

    # build reference configuration
    nNd = nEl + p
    X0 = np.linspace(0, L, nNd)
    Xi = uniform_knot_vector(p, nEl)
    for i in range(nNd):
        X0[i] = np.sum(Xi[i+1:i+p+1])
    X0 = X0 * L / p
    Y0 = np.zeros_like(X0)
    Q = np.hstack((X0, Y0))
    u0 = np.zeros_like(Q)

    q0 = np.hstack((X0, Y0))

    beam = Euler_bernoulli2D(A_rho0, material_model, p, nEl, nQP, Q=Q, q0=q0, u0=u0)

    # np.set_printoptions(precision=1)
    # model = Model()
    # model.add(beam)
    # model.assemble()

    # f_int = model.f_pot(0, model.q0)
    # print(f'f_int:{f_int}')

    # f_int_q = model.f_pot_q(0, model.q0).toarray()
    # print(f'f_int_q:\n{f_int_q}')

    # exit()

    # left joint
    r_OB1 = np.zeros(3)
    frame_left = Frame(r_OP=r_OB1)
    joint_left = Spherical_joint(frame_left, beam, r_OB1, frame_ID2=(0,))

    # omega = 2 * np.pi / 2
    # A = -1
    # r_OB1 = lambda t: np.array([0, 0, A * np.sin(omega * t)])
    # r_OB1_t = lambda t: np.array([0, 0, A * omega * np.cos(omega * t)])
    # r_OB1_tt = lambda t: np.array([0, 0, -A * omega**2 * np.sin(omega * t)])
    # frame_left = Frame(r_OP=r_OB1, r_OP_t=r_OB1_t, r_OP_tt=r_OB1_tt)
    # joint_left = Spherical_joint(frame_left, beam, r_OB1(0), frame_ID2=(0,))

    # right joint
    r_OB2 = np.array([L, 0, 0])
    frame_right = Frame(r_OP=r_OB2)
    joint_right = Spherical_joint(beam, frame_right, r_OB2, frame_ID1=(1,))

    # gravity
    g = np.array([0, - A_rho0 * L * 9.81, 0]) * 1.0e-6
    if statics:
        f_g = Line_force(lambda xi, t: t * g, beam)
    else:
        f_g = Line_force(lambda xi, t: g, beam)

    # assemble the model
    model = Model()
    model.add(beam)
    model.add(frame_left)
    model.add(joint_left)
    model.add(frame_right)
    model.add(joint_right)
    model.add(f_g)
    model.assemble()

    if statics:
        solver = Newton(model, n_load_stepts=10, max_iter=10, numerical_jacobian=False)
        # solver = Newton(model, n_load_stepts=50, max_iter=10, numerical_jacobian=True)
        sol = solver.solve()
        t = sol.t
        q = sol.q
        # print(f'pot(t0, q0): {model.E_pot(t[0], q[0])}')
        # print(f'pot(t1, q1): {model.E_pot(t[-1], q[-1])}')
        # exit()
    else:
        t0 = 0
        t1 = 4
        dt = 1e-3
        # solver = Euler_backward(model, t1, dt, numerical_jacobian=False, debug=False)
        solver = Moreau(model, t1, dt)
        # solver = Moreau_sym(model, t1, dt)
        # solver = Generalized_alpha_1(model, t1, dt, rho_inf=0.75)
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK23')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='RK45')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='DOP853')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='Radau')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='BDF')
        # solver = Scipy_ivp(model, t1, dt, atol=1.e-6, method='LSODA')

        sol = solver.solve()

        # from cardillo.solver import save_solution
        # save_solution(sol, f'test')

        # from cardillo.solver import load_solution
        # sol = load_solution(f'test')

        t = sol.t
        q = sol.q
        # t, q, u, la_g, la_gamma = sol.unpack()

    # animate configurations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    scale = L
    ax.set_xlim3d(left=0, right=L)
    ax.set_ylim3d(bottom=-L/2, top=L/2)
    ax.set_zlim3d(bottom=-L/2, top=L/2)

    # prepare data for animation
    if statics:
        frames = len(t)
        interval = 100
    else:
        frames = len(t)
        target_frames = 100
        frac = int(frames / target_frames)
        animation_time = 1
        interval = animation_time * 1000 / target_frames

        frames = target_frames
        t = t[::frac]
        q = q[::frac]
    
    x0, y0 = q0.reshape((2, -1))
    z0 = np.zeros_like(x0)
    center_line0, = ax.plot(x0, y0, z0, '-ok')

    x1, y1 = q[-1].reshape((2, -1))
    z1 = np.zeros_like(x1)
    center_line, = ax.plot(x1, y1, z1, '-ob')

    plt.show()
    exit()

    def update(t, q, center_line):
        print(f'q.shape: {q.shape}')
        x, y = q[-1].reshape((2, -1))
        z = np.zeros_like(x)
        center_line.set_data(x, y)
        center_line.set_3d_properties(z)

        return center_line,

    def animate(i):
        update(t[i], q[i], center_line)

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
    plt.show()