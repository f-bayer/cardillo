import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import pickle
import pathlib
import datetime
from cardillo.discretization.mesh3D import Mesh3D, cube
from cardillo.discretization.mesh2D import Mesh2D, rectangle
from cardillo.discretization.B_spline import Knot_vector, fit_B_spline_volume
from cardillo.discretization.indexing import flat3D, flat2D
from cardillo.model.continuum import Second_gradient, Pantobox_beam_network
from cardillo.solver import Newton, Euler_backward, Generalized_alpha_1
from cardillo.model import Model
from cardillo.math.algebra import A_IK_basic_z
from cardillo.model.force_distr2D import Force_distr2D
from cardillo.model.force_distr3D import Force_distr3D
from cardillo.model.bilateral_constraints.implicit.incompressibility import Incompressibility

def save_solution(sol, filename):
    import pickle
    with open(filename, mode='wb') as f:
        pickle.dump(sol, f)

def test_cube():

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / 'results' / f"{file_name}_cube_2x2x6_ext" / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_path = file_path.parent / 'sol'

    TractionForce = False
    Gravity = False
    Statics = True
    Incompressible = False
    save_sol = True
    torsion = False
    tension = True

    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 6)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=2, basis='B-spline', nq_n=3)



    # material model
    l = 70.0  # in mm
    L = 3 * l
    a = 1.0  # in mm
    b = 1.0  # in mm
    Yb = 50.0  # in GPa
    Gb = Yb / (2 + 0.8)
    rp = 0.45 # in mm
    hp = 1.5 # in mm
    Jn = a**3*b/12
    Jg = a*b**3/12
    Jt = 0.196*a**3*b
    nf = 10
    p = l/np.sqrt(2)/nf
    Ke = Yb*a*b/p
    Kg = Yb*Jg/p
    Kn = Yb*Jn/p
    Kt = Gb*Jt/p
    Kp = Gb*np.pi*rp**4/2/hp/p**2
    Ks = Kp
    Kc = Ks
    # H = L/3
    nsH = nf/l
    # nsH = 1

    # reference configuration is a cube
    W = L/3
    B = L/3
    H = L
    cube_shape = (W, B, H)
    Z = cube(cube_shape, mesh, Greville=False)

    mat = Pantobox_beam_network(Ke*nsH, Ks*nsH, Kg*nsH, Kn*nsH, Kt*nsH, Kc*nsH)

    density = 1.0e-3

    if Statics:
        # boundary conditions
        if TractionForce:
            # cDOF = mesh.surface_qDOF[0].reshape(-1)
            cDOF = mesh.surface_qDOF[2].reshape(-1)
            # cDOF = mesh.surface_qDOF[4].reshape(-1)
            b = lambda t: Z[cDOF]

        else:
            if tension:
                cDOF1 = mesh.surface_qDOF[4].ravel()
                cDOF3 = mesh.surface_qDOF[5][0]
                cDOF4 = mesh.surface_qDOF[5][1]
                cDOF2 = mesh.surface_qDOF[5][2]
                cDOF134 = np.concatenate((cDOF1, cDOF3, cDOF4,))
                cDOF = np.concatenate((cDOF134, cDOF2))
                b1 = lambda t: Z[cDOF134]
                b2 = lambda t: Z[cDOF2] + t * 50.0
                b = lambda t: np.concatenate((b1(t), b2(t)))
            if torsion:
                cDOF1 = mesh.surface_qDOF[4].ravel()
                cDOF3 = mesh.surface_qDOF[5][0]
                cDOF4 = mesh.surface_qDOF[5][1]
                cDOF2 = mesh.surface_qDOF[5].ravel()
                cDOF = np.concatenate((cDOF1, cDOF2))

                def bt(t, phi0=0.5*np.pi, h=100):
                    cDOF2_xyz = cDOF2.reshape(3, -1).T
                    out = np.zeros_like(Z)

                    phi = t * phi0
                    R = A_IK_basic_z(phi)

                    th = t * np.array([0, 0, h])
                    for DOF in cDOF2_xyz:
                        out[DOF] = R @ (Z[DOF] - [W/2, B/2, 0]) + th + [W/2, B/2, 0]
                    
                    return out[cDOF2]

                b1 = lambda t: Z[cDOF1]
                b = lambda t: np.concatenate((b1(t), bt(t)))
            # cDOF = mesh.surface_qDOF[4].ravel()
            # b = lambda t: Z[cDOF]
    else:
        cDOF_xi = mesh.surface_qDOF[4][0]
        cDOF_eta = mesh.surface_qDOF[4][1]
        cDOF_zeta = mesh.surface_qDOF[4][2]
        cDOF = np.concatenate((cDOF_xi, cDOF_eta, cDOF_zeta))
        Omega = 2 * np.pi
        b_xi = lambda t: Z[cDOF_xi] + 0.1 * np.sin(Omega * t)
        b_eta = lambda t: Z[cDOF_eta]
        b_zeta = lambda t: Z[cDOF_zeta]
        b = lambda t: np.concatenate((b_xi(t), b_eta(t), b_zeta(t)))


    # 3D continuum
    continuum = Second_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)
    # continuum = First_gradient(density, mat, mesh, Z)


    # build model
    model = Model()
    model.add(continuum)

    if Incompressible:
        incompressibility = Incompressibility(continuum, la_mesh)
        model.add(incompressibility)

    if TractionForce:
        # F = lambda t, xi, eta: t * np.array([-2.5e0, 0, 0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        # model.add(Force_distr2D(F, continuum, 1))
        # F = lambda t, xi, eta: t * np.array([0, -2.5e0, 0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        # model.add(Force_distr2D(F, continuum, 5))
        F = lambda t, xi, eta: np.array([0, 0, -5e0]) * (0.25 - (xi-0.5)**2) * (0.25 - (eta-0.5)**2)
        model.add(Force_distr2D(F, continuum, 5))
    
    if Gravity:
        if Statics:
            G = lambda t, xi, eta, zeta: t * np.array([0, 0, -9.81 * density])
        else:
            G = lambda t, xi, eta, zeta: np.array([0, 0, -9.81 * density])
        model.add(Force_distr3D(G, continuum))

    model.assemble()

    # M = model.M(0, model.q0)
    # np.set_printoptions(precision=5, suppress=True)
    # print(M.toarray())
    # print(np.linalg.det(M.toarray()))
    

    if Statics:
    # static solver
        n_load_steps = 20
        tol = 1.0e-5
        max_iter = 10
        solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)

    else:
        t1 = 10
        dt = 1e-1
        # solver = Generalized_alpha_1(model, t1, dt=dt, variable_dt=False, rho_inf=0.25)
        solver = Euler_backward(model, t1, dt)


    if save_sol:

        # import cProfile, pstats
        # pr = cProfile.Profile()
        # pr.enable()
        sol = solver.solve()
        # pr.disable()

        # sortby = 'cumulative'
        # ps = pstats.Stats(pr).sort_stats(sortby)
        # ps.print_stats(0.1) # print only first 10% of the list
            # export solution object
            # if not os.path.exists(export_dir):
            #     os.makedirs(export_dir)
        save_solution(sol, str(export_path))
    else:
        sol = pickle.load( open(str(export_path), 'rb') )

    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(*model.contributions[0].z(sol.t[-1], sol.q[-1]).reshape(3,-1))
    # z = model.contributions[0].z(sol.t[-1], sol.q[-1])
    # model.contributions[0].F_qp(z)
    # F = model.contributions[0].F
    # J = list(map(np.linalg.det, F.reshape(-1,3,3)))
    # plt.figure()
    # plt.plot(np.array(J))

    # z_y = z[mesh.nn:2*mesh.nn:mesh.nn_xi*mesh.nn_eta]

    # plt.figure()
    # from scipy.interpolate import BSpline
    # Spline = BSpline(Zeta.data, z_y, 2)

    # xx = np.linspace(0,1,20)

    # plt.plot(xx,Spline(xx))

    # plt.show()
    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # sol = solver.solve()
    # pr.disable()

    # vtk export
    # continuum.post_processing(sol.t, sol.q, 'cube_splines_incomp')
    # continuum.post_processing(sol.t, sol.q, filepath.parent / filepath.stem)
    continuum.post_processing(sol.t, sol.q, file_path, binary=True)

def test_cylinder():  
    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / 'results' / f"{file_name}_cylinder" / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # export_path = file_path.parent / 'sol'

    # build mesh
    degrees = (3, 3, 3)
    QP_shape = (3, 3, 3)
    element_shape = (2, 2, 4)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    Zeta = Knot_vector(degrees[2], element_shape[2])
    knot_vectors = (Xi, Eta, Zeta)
    
    mesh = Mesh3D(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=3)
    
    def cylinder(xi, eta, zeta, R=1, H=3):
        xi_ = 2 * xi - 1
        eta_ = 2 * eta - 1

        if np.abs(xi_) > np.abs(eta_):
            r = np.sqrt(1 + eta_**2)
        else:
            r = np.sqrt(1 + xi_**2)

        x = R / r * xi_
        y = R / r * eta_
        z = zeta * H
        return x, y, z

    nxi, neta, nzeta = 10, 10, 10
    xi = np.linspace(0, 1, num=nxi)
    eta = np.linspace(0, 1, num=neta)
    zeta = np.linspace(0, 1, num=nzeta)
    
    n3 = nxi * neta * nzeta
    knots = np.zeros((n3, 3))
    Pw = np.zeros((n3, 3))
    for i, xii in enumerate(xi):
        for j, etai in enumerate(eta):
            for k, zetai in enumerate(zeta):
                idx = flat3D(i, j, k, (nxi, neta, nzeta))
                knots[idx] = xii, etai, zetai
                Pw[idx] = cylinder(xii, etai, zetai)

    cDOF_ = np.array([], dtype=int)
    qc = np.array([], dtype=float).reshape((0, 3))
    X, Y, Z_ = fit_B_spline_volume(mesh, knots, Pw, qc, cDOF_)
    Z = np.concatenate((X, Y, Z_))

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2)

    # boundary conditions
    cDOF1 = mesh.surface_qDOF[0].reshape(-1)
    cDOF2 = mesh.surface_qDOF[1].reshape(-1)
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]

    def b2(t, phi0=np.pi, h=0.25):
        cDOF2_xyz = cDOF2.reshape(3, -1).T
        out = np.zeros_like(Z)

        phi = t * phi0
        R = A_IK_basic_z(phi)

        th = t * np.array([0, 0, h])
        for DOF in cDOF2_xyz:
            out[DOF] = R @ Z[DOF] + th
        
        return out[cDOF2]

    b = lambda t: np.concatenate((b1(t), b2(t)))

    density = 1e-2

    # 3D continuum
    continuum = First_gradient(density, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 10
    tol = 1.0e-5
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    import cProfile, pstats
    pr = cProfile.Profile()
    pr.enable()
    sol = solver.solve()
    pr.disable()

    sortby = 'cumulative'
    ps = pstats.Stats(pr).sort_stats(sortby)
    ps.print_stats(0.1) # print only first 10% of the list

    # vtk export
    continuum.post_processing(sol.t, sol.q, file_path)

def test_rectangle():

    file_name = pathlib.Path(__file__).stem
    file_path = pathlib.Path(__file__).parent / 'results' / f"{file_name}_rectangle" / file_name
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # export_path = file_path.parent / 'sol'  

    # build mesh
    degrees = (1, 1)
    QP_shape = (3, 3)
    element_shape = (4, 8)

    Xi = Knot_vector(degrees[0], element_shape[0])
    Eta = Knot_vector(degrees[1], element_shape[1])
    knot_vectors = (Xi, Eta)
    
    mesh = Mesh2D(knot_vectors, QP_shape, derivative_order=1, basis='B-spline', nq_n=2)

    # reference configuration is a cube
    L = 2
    B = 4

    rectangle_shape = (L, B)
    Z = rectangle(rectangle_shape, mesh, Greville=True)

    # material model    
    mu1 = 0.3
    mu2 = 0.5
    mat = Ogden1997_compressible(mu1, mu2, dim=2)

    # boundary conditions
    cDOF1 = mesh.edge_qDOF[0].reshape(-1)
    cDOF2 = mesh.edge_qDOF[1][1]
    cDOF = np.concatenate((cDOF1, cDOF2))
    b1 = lambda t: Z[cDOF1]
    b2 = lambda t: Z[cDOF2] + t * 4
    b = lambda t: np.concatenate((b1(t), b2(t)))

    # 3D continuum
    continuum = First_gradient(1, mat, mesh, Z, z0=Z, cDOF=cDOF, b=b)

    # vtk export reference configuration
    # continuum.post_processing_single_configuration(0, Z, 'rectangleReferenceConfig.vtu')

    # build model
    model = Model()
    model.add(continuum)
    model.assemble()

    # static solver
    n_load_steps = 30
    tol = 1.0e-6
    max_iter = 10
    solver = Newton(model, n_load_steps=n_load_steps, tol=tol, max_iter=max_iter)
    
    sol = solver.solve()

    # # vtk export
    continuum.post_processing(sol.t, sol.q, file_path)



def write_xml():
    # write paraview PVD file, see https://www.paraview.org/Wiki/ParaView/Data_formats#PVD_File_Format
    from xml.dom import minidom
    
    root = minidom.Document()
    
    vkt_file = root.createElement('VTKFile')
    vkt_file.setAttribute('type', 'Collection')
    root.appendChild(vkt_file)
    
    collection = root.createElement('Collection')
    vkt_file.appendChild(collection)

    for i in range(10):
        ti = 0.1 * i
        dataset = root.createElement('DataSet')
        dataset.setAttribute('timestep', f'{ti:0.6f}')
        # dataset.setAttribute('group', '')
        # dataset.setAttribute('part', '0')
        dataset.setAttribute('file', f'continuum{i}.vtu')
        collection.appendChild(dataset)
 
    xml_str = root.toprettyxml(indent ="\t")   
    save_path_file = "continuum.pvd"
    with open(save_path_file, "w") as f:
        f.write(xml_str)

    
if __name__ == "__main__":
    test_cube()
    # test_cylinder()
    # test_rectangle()
    # write_xml()
