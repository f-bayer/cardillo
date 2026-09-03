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
from cardillo.discrete import RigidBody, Box
from cardillo.constraints import Prismatic
from cardillo.force_laws import Spring, KelvinVoigtElement
from cardillo.interactions import TwoPointInteraction
from cardillo.forces import Force

from cardillo.solver.skhippr import (
    SkhipprStaticContinuation,
    SkhipprHBM,
    SkhipprHBMContinuation,
)


def main(
    static_continuation=True,
    hbm_single=True,
    hbm_continuation=True,
    compliance_form=True,
    export_vtk=False,
    show_plots=True,
):
    # parameters
    a = 1.0
    mass = 1.0
    l0 = 1.2
    k_spring = 3.0
    d_damper = 0.5

    # build system
    cardillo_system = System()

    # rigid body + prismatic constraint allowing for motion along ex_I
    rb = Box(RigidBody)([0.1, 0.1, 0.1], mass=mass, B_Theta_C=np.eye(3))
    con = Prismatic(cardillo_system.origin, rb, axis=0)

    # spring
    inter_spring = TwoPointInteraction(
        cardillo_system.origin, rb, B_r_CP1=np.array([0, 0, a])
    )
    spring = Spring(inter_spring, k=k_spring, l_ref=l0, compliance_form=compliance_form)

    # damper
    inter_damper = TwoPointInteraction(
        cardillo_system.origin, rb, B_r_CP1=np.array([2 * a, 0, 0])
    )
    # damping force always with compliance_form=False, as k is 0!
    damper = KelvinVoigtElement(inter_damper, k=0.0, d=d_damper, compliance_form=False)

    # force (here [0, 0, 0], later updated to what is required for static_continuation, hbm_single or hbm_continuation)
    force = Force(np.zeros(3), rb)

    # add to system and assemble
    cardillo_system.add(rb, con, spring, damper, force)
    cardillo_system.assemble()

    if static_continuation:
        # specify force
        force.force = lambda t: np.array([-0.5 + t, 0.0, 0.0])

        # solver
        cardillo_solver = SkhipprStaticContinuation(cardillo_system)
        sol, skhippr_sol = cardillo_solver.solve()

        if export_vtk:
            # vtk-export
            dir_name = Path(__file__).parent
            cardillo_system.export(dir_name, "vtk_static_continuation", sol)

        # SKHiPPr plot continuation
        plot_continuation(skhippr_sol)
        if show_plots:
            plt.show()

    if hbm_single:
        # specify force
        omega = 0.01
        force.force = lambda t: np.array([0.5 * np.sin(omega * t) - 0.5, 0, 0])

        # solver
        cardillo_solver = SkhipprHBM(cardillo_system, omega, newton_max_iter=40)
        sol, skhippr_sol = cardillo_solver.solve()

        if export_vtk:
            # vtk-export
            dir_name = Path(__file__).parent
            cardillo_system.export(dir_name, "vtk_hbm", sol)

        # SKHiPPR plots
        plot_period(skhippr_sol, title=f"initial periodic solution")
        plot_phase(
            skhippr_sol,
            idx=[0, cardillo_system.nq],
            title=f"initial periodic solution",
        )
        if show_plots:
            plt.show()

    if hbm_continuation:
        # specify force via update function for skhippr continuation
        omega = 0.01

        def set_parameter(F):
            force.force = lambda t, F=F: np.array([F * np.sin(omega * t) - 0.5, 0, 0])

        # solver
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

        if export_vtk:
            # vtk-export
            dir_name = Path(__file__).parent
            solutions_export = solutions[:: int(np.max([1, len(solutions) / 10]))]
            for i, sol in enumerate(solutions_export):
                # same parameter value can exist in multiple solutions (continuation)!
                cardillo_system.export(
                    dir_name, f"vtk_hbm_{i}__param_{sol.param:.4f}", sol
                )

        # SKHiPPR plots
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
        if show_plots:
            plt.show()


if __name__ == "__main__":
    main(False, True, False, compliance_form=True, show_plots=False)
    main(False, True, False, compliance_form=False, show_plots=False)
    plt.show()
