"""Single DFG 2D-1 run: Newton history, drag and lift, flow fields.

The fields are written for ParaView (``results/dfg2d1/single_run.pvd``, on a
fitted reparametrization of the fluid domain) and plotted from that file
with PyVista: velocity magnitude with streamlines, and pressure. The plot is
saved to ``figures/dfg2d1/single_run.png``; ``--show`` opens it in a window.

    python experiments/dfg2d1/single_run.py
    python experiments/dfg2d1/single_run.py --plot-only --show
"""

import sys

from cutfem import navier_stokes as ns
from cutfem import study
from cutfem.problems import dfg2d1

PARAMS = dfg2d1.Params(n_y=64)
VTK_PATH = study.RESULTS_DIR / "dfg2d1" / "single_run.pvd"
FIGURE_PATH = study.FIGURES_DIR / "dfg2d1" / "single_run.png"


def compute():
    """Solve once, print the benchmark quantities and write the fields."""
    study.print0(PARAMS)
    with dfg2d1.build(PARAMS) as system:
        study.print0(f"h = {system.h:.5f}, n_dofs = {system.n_dofs}")
        solution = dfg2d1.solve(system, verbose=True)
        values = dfg2d1.benchmark(system, solution)
        errors = dfg2d1.errors(system, solution)
        reynolds = ns.cell_reynolds(system, solution[0])
        history = list(system.history)
        dfg2d1.save_vtk(VTK_PATH, system, solution)

    for key, value in values.items():
        study.print0(f"  {key} = {value:.8f}   (reference "
                     f"{dfg2d1.REFERENCE[key]:.8f}, rel. error {errors[f'err_{key}']:.2e})")
    study.print0(f"  max cell Reynolds number = {reynolds:.2f}")
    study.print0(f"Saved {VTK_PATH.relative_to(study.ROOT)}")
    return {"params": PARAMS, "values": values, "errors": errors,
            "history": history, "cell_reynolds": reynolds}


def show_fields(path, show=False):
    """Plot |u| with streamlines and p from the ParaView file, with PyVista."""
    import numpy as np
    import pyvista as pv

    grid = pv.get_reader(str(path)).read()
    while isinstance(grid, pv.MultiBlock):             # .pvd -> time step -> piece
        grid = grid[0]
    grid = grid.extract_surface(nonlinear_subdivision=3,   # Lagrange cells -> triangles
                                algorithm="dataset_surface")
    u = grid.point_data["uh"]
    grid.point_data["|u|"] = np.linalg.norm(u, axis=1)
    if u.shape[1] == 2:                                 # streamlines need 3D vectors
        grid.point_data["uh"] = np.column_stack([u, np.zeros(len(u))])

    seeds = pv.Line((0.01, 0.01, 0.0), (0.01, PARAMS.height - 0.01, 0.0), resolution=24)
    lines = grid.streamlines_from_source(seeds, vectors="uh", max_length=5.0,
                                         max_steps=50_000,
                                         integration_direction="forward")

    bar = dict(vertical=True, height=0.7, width=0.04, position_x=0.9,
               position_y=0.15, title_font_size=16, label_font_size=13,
               n_labels=5, fmt="%.2f")
    plotter = pv.Plotter(shape=(2, 1), off_screen=not show, window_size=(1800, 800))
    plotter.subplot(0, 0)
    plotter.add_text("velocity magnitude", font_size=11)
    plotter.add_mesh(grid, scalars="|u|", cmap="viridis",
                     scalar_bar_args={**bar, "title": "|u|"})
    if lines.n_points:
        plotter.add_mesh(lines, color="white", line_width=1.0, opacity=0.6)
    plotter.view_xy()
    plotter.subplot(1, 0)
    plotter.add_text("pressure", font_size=11)
    plotter.add_mesh(grid, scalars="ph", cmap="coolwarm",
                     scalar_bar_args={**bar, "title": "p", "fmt": "%.3f"})
    plotter.view_xy()
    plotter.link_views()
    plotter.camera.zoom(2.0)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(FIGURE_PATH), auto_close=True)
    study.print0(f"Saved {FIGURE_PATH.relative_to(study.ROOT)}")


if __name__ == "__main__":
    study.run("dfg2d1/single_run", compute)
    show_fields(VTK_PATH, show="--show" in sys.argv)
