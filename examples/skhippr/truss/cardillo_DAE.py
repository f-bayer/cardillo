"""
Find and plot the equilibrium of the Truss system using SKHiPPR.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# --- Visualization ---
from skhippr.visualization.continuation import plot_continuation
from skhippr.visualization.equilibria import plot_equilibrium, plot_eigenvalues
from skhippr.visualization.cycles import plot_period, plot_phase, animate_phase


from cardillo import System
from cardillo.discrete import RigidBody, PointMass, Box
from cardillo.constraints import Prismatic
from cardillo.constraints._base import ProjectedPositionOrientationBase
from cardillo.force_laws import Spring, KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force

from cardillo.solver.skhippr import (
    SkhipprStaticContinuation,
    SkhipprHBM,
    SkhipprHBMContinuation,
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

    rb = Box(RigidBody)([0.1, 0.1, 0.1], mass=mass, B_Theta_C=np.eye(3))
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
    static_continuation = True
    static_continuation = False

    hbm = True
    hbm = False

    if static_continuation:
        force = Force(lambda t: np.array([-0.5 + t, 0.0, 0.0]), rb)
    else:
        omega = 2 * np.pi
        omega = 0.01
        force = Force(lambda t: np.array([0.5 * np.sin(omega * t) - 0.5, 0, 0]), rb)

    cardillo_system.add(rb, con, spring, damper, force)

    cardillo_system.assemble()

    if static_continuation:
        cardillo_solver = SkhipprStaticContinuation(cardillo_system)

    elif hbm:
        cardillo_solver = SkhipprHBM(cardillo_system, omega)

        sol, skhippr_sol = cardillo_solver.solve()
        print(skhippr_sol.eigenvalues)

        # vtk-export
        dir_name = Path(__file__).parent
        cardillo_system.export(dir_name, "vtk", sol)

        if static_continuation:
            plot_continuation(skhippr_sol)
        else:
            plot_period(skhippr_sol, title=f"initial periodic solution")
            plot_phase(
                skhippr_sol,
                idx=[0, cardillo_system.nq],
                title=f"initial periodic solution",
            )

        plt.show()

    # continuation with HBM
    # update function for skhippr continuation
    def set_parameter(F):
        force.force = lambda t, F=F: np.array([F * np.sin(omega * t) - 0.5, 0, 0])

    solver = SkhipprHBMContinuation(
        cardillo_system,
        omega,
        set_parameter,
        0.0,
        1.0,
        max_step_size=0.03,
        compute_stability=True,
    )
    solutions, branch = solver.solve()

    def plot_fun(bp):
        # maximum of q[0] over one period
        return (bp.equations[0].param, np.max(bp.equations[0].x_time()[0, :]))

    plot_continuation(
        branch,
        marker="x",
        plot_fun=plot_fun,
        xlabel="parameter (F)",
        ylabel="amplitude of q[0]",
    )

    _ = animate_phase(branch, idx=[0, cardillo_system.nq], interval=60)

    plt.show()


if __name__ == "__main__":
    main()
