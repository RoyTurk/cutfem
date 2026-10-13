"""Step 1: pressure-driven Navier-Stokes flow through the gyroid (single run).

Reports the problem size, Newton history, wall times (mesh/forms, solve),
peak memory, flow rates and mass balance, superficial velocity, apparent
permeability and Reynolds numbers. Writes velocity and pressure for ParaView
(``results/gyroid/single_run.pvd``), and from that file:

* ``figures/gyroid/single_run.png``: |u| on the fluid boundary (PyVista, 3D);
* ``figures/gyroid/single_run_plane.pdf``: |u| and p on a mid-height plane;
* ``figures/gyroid/single_run_sections.pdf``: u_x on four cross-sections;
* ``figures/gyroid/single_run_profiles.pdf``: mean flow rate and pressure of
  each cell layer along the channel, from exact cut integrals of the
  finite-element solution (``gyroid.layer_profiles``), not from the VTK file
  (its interpolated velocity integrates ~2 % low).

Slices are taken from the tessellated mesh (slicing the curved cells directly
is inaccurate) on planes at background cell centres (a plane on a grid plane
is degenerate).

    python experiments/gyroid/single_run.py
    python experiments/gyroid/single_run.py --plot-only
"""

import resource
import sys
import time

import numpy as np

from cutfem import navier_stokes as ns
from cutfem import study
from cutfem.problems import gyroid

PARAMS = gyroid.Params(n_cell=8)
VTK_PATH = study.RESULTS_DIR / "gyroid" / "single_run.pvd"
FIGURE_PATH = study.FIGURES_DIR / "gyroid" / "single_run.png"
SECTIONS_X = (0.5, 1.0, 1.5, 2.5)           # cross-sections, in units of L (snapped)


def peak_memory_gb():
    """Peak resident memory of this process (macOS: bytes, Linux: kB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def compute():
    """Build, solve, report and write the fields."""
    study.print0(PARAMS)
    start = time.perf_counter()
    system = gyroid.build(PARAMS)
    t_build = time.perf_counter() - start
    n_dofs = system.n_dofs
    study.print0(f"h = {system.h:.4f}, unknowns = {n_dofs}, "
                 f"build (mesh, quadrature, forms) {t_build:.1f} s")
    try:
        start = time.perf_counter()
        solution = gyroid.solve(system, verbose=True)
        t_solve = time.perf_counter() - start
        values = gyroid.benchmark(system, solution)
        profiles = gyroid.layer_profiles(system, solution)
        reynolds_cell = ns.cell_reynolds(system, solution[0])
        history = list(system.history)
        start = time.perf_counter()
        gyroid.save_vtk(VTK_PATH, system, solution)
        t_vtk = time.perf_counter() - start
    finally:
        system.destroy()

    study.print0(f"solve {t_solve:.1f} s ({len(history) - 1} Newton steps), "
                 f"VTK {t_vtk:.1f} s, peak memory {peak_memory_gb():.2f} GB")
    for key, value in values.items():
        study.print0(f"  {key:>13} = {value:.6e}")
    study.print0(f"  max cell Reynolds number = {reynolds_cell:.2f}")
    study.print0(f"Saved {VTK_PATH.relative_to(study.ROOT)}")
    return {"params": PARAMS, "n_dofs": n_dofs, "values": values,
            "history": history, "cell_reynolds": reynolds_cell, "profiles": profiles,
            "t_build": t_build, "t_solve": t_solve, "peak_memory_gb": peak_memory_gb()}


def read_fields(path):
    """Velocity, |u|, u_x and pressure on the fluid mesh, tessellated.

    The reparametrized mesh has curved (degree-3 Lagrange) cells; VTK slices
    them inaccurately (cross-sections 3-6 % too small or large, with gaps).
    Tessellating into linear cells first makes the slices exact.
    """
    import pyvista as pv

    grid = pv.get_reader(str(path)).read()
    while isinstance(grid, pv.MultiBlock):
        grid = grid[0]
    grid = grid.tessellate(max_n_subdivide=3)
    u = grid.point_data["uh"]
    grid.point_data["|u|"] = np.linalg.norm(u, axis=1)
    grid.point_data["u_x"] = u[:, 0]
    return grid


def off_grid(value, params=PARAMS):
    """Nearest background cell centre: a plane there never lies on cell faces."""
    h = params.L / params.n_cell
    return (np.floor(value / h) + 0.5) * h


def show_fields(path, show=False):
    """|u| on the boundary of the fluid domain (PyVista, 3D view).

    (Streamlines are left to ParaView: PyVista's tracer does not follow the
    high-order cells of the reparametrized mesh.)
    """
    import pyvista as pv

    grid = read_fields(path)
    L, length = PARAMS.L, PARAMS.N * PARAMS.L
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(off_screen=not show, window_size=(1400, 800))
    plotter.add_text("velocity magnitude |u| on the fluid domain", font_size=10)
    plotter.add_mesh(surface, scalars="|u|", cmap="viridis",
                     clim=[0.0, float(grid.point_data["|u|"].max())],
                     scalar_bar_args=dict(title="|u|", vertical=False, width=0.5,
                                          position_x=0.25, position_y=0.05, fmt="%.2f"))
    plotter.add_mesh(pv.Box(bounds=(0, length, 0, L, 0, L)), style="wireframe",
                     color="black", line_width=1)
    plotter.view_isometric()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(FIGURE_PATH), auto_close=True)
    study.print0(f"Saved {FIGURE_PATH.relative_to(study.ROOT)}")


def sample_plane(grid, normal, offset, resolution):
    """Fields on a regular grid of the plane ``normal = offset``.

    Points outside the fluid are masked. Sampling (VTK probe filter) avoids
    triangulating the slice, whose degenerate triangles smear across plots.
    Returns ``(extent, {name: masked 2D array})`` in the in-plane coordinates
    (x, y) for a z-plane and (y, z) for an x-plane.
    """
    import pyvista as pv

    L, length = PARAMS.L, PARAMS.N * PARAMS.L
    in_plane = [i for i in range(3) if i != "xyz".index(normal)]
    sizes = np.array([length, L, L])[in_plane]
    n = np.maximum(np.round(sizes / sizes.min() * resolution).astype(int), 2)
    a, b = [(np.arange(m) + 0.5) / m * size for m, size in zip(n, sizes, strict=True)]
    A, B = np.meshgrid(a, b, indexing="xy")
    points = np.zeros((A.size, 3))
    points[:, in_plane[0]], points[:, in_plane[1]] = A.ravel(), B.ravel()
    points[:, "xyz".index(normal)] = offset
    probe = pv.PolyData(points).sample(grid)
    outside = probe.point_data["vtkValidPointMask"].reshape(A.shape) == 0
    fields = {key: np.ma.masked_where(outside, probe.point_data[key].reshape(A.shape))
              for key in ("|u|", "ph", "u_x")}
    return (0.0, sizes[0], 0.0, sizes[1]), fields


def plot(data):
    """Matplotlib views of the slices: the mid-height plane and cross-sections."""
    from cutfem import plotting

    grid = read_fields(VTK_PATH)
    L = PARAMS.L
    u_max = float(grid.point_data["|u|"].max())
    image = dict(origin="lower", interpolation="nearest", rasterized=True)

    def colormap(name):
        """Colour map with the solid (masked) region in light grey."""
        import matplotlib

        cmap = matplotlib.colormaps[name].copy()
        cmap.set_bad("0.85")
        return cmap

    # |u| and p on the plane z = const
    z_mid = off_grid(0.5 * L)
    extent, fields = sample_plane(grid, "z", z_mid, resolution=300)
    fig_plane, axes = plotting.figure(nrows=2, ncols=1, aspect=0.42, sharex=True)
    for ax, key, cmap, clim, label in (
            (axes[0], "|u|", "viridis", (0.0, u_max), r"$|u_h|$"),
            (axes[1], "ph", "coolwarm", (0.0, PARAMS.p_in), r"$p_h$")):
        shown = ax.imshow(fields[key], extent=extent, cmap=colormap(cmap), vmin=clim[0],
                          vmax=clim[1], **image)
        fig_plane.colorbar(shown, ax=ax, label=label, pad=0.02)
        ax.set_ylabel(r"$y$")
        ax.grid(False)
    axes[1].set_xlabel(r"$x$")
    plotting.panel_label(axes[0], rf"(a) velocity magnitude, $z = {z_mid:.4g}$")
    plotting.panel_label(axes[1], rf"(b) pressure, $z = {z_mid:.4g}$")

    # u_x on cross-sections x = const
    fig_sections, axes = plotting.figure(nrows=2, ncols=2, aspect=1.0,
                                         sharex=True, sharey=True)
    for k, (ax, x) in enumerate(zip(axes.flat, SECTIONS_X, strict=True)):
        x_section = off_grid(x * L)
        extent, fields = sample_plane(grid, "x", x_section, resolution=300)
        shown = ax.imshow(fields["u_x"], extent=extent, cmap=colormap("RdBu_r"),
                          vmin=-u_max, vmax=u_max, **image)
        ax.grid(False)
        plotting.panel_label(ax, rf"({'abcd'[k]}) $x = {x_section:.4g}$")
    for ax in axes[1]:
        ax.set_xlabel(r"$y$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z$")
    fig_sections.colorbar(shown, ax=axes, label=r"$u_{h,x}$", shrink=0.8)

    # Flow rate and mean pressure along the channel (finite-element layer
    # integrals, gyroid.layer_profiles; not from the VTK file)
    profiles, Q_out = data["profiles"], data["values"]["Q_out"]
    fig_profiles, axes = plotting.figure(nrows=2, ncols=1, width=0.8, aspect=0.4,
                                         sharex=True)
    x, length = profiles["x"], PARAMS.N * L
    axes[0].plot(x, np.asarray(profiles["Q"]) / Q_out, marker="o", markersize=3)
    axes[0].axhline(1.0, color=plotting.INK, linestyle="--", linewidth=0.8)
    axes[0].set_ylabel(r"$\bar Q / Q_{\mathrm{out}}$")
    axes[1].plot(x, profiles["p_mean"], marker="o", markersize=3)
    axes[1].plot([0, length], [PARAMS.p_in, 0.0], color=plotting.INK, linestyle="--",
                 linewidth=0.8)
    axes[1].set_ylabel(r"$\bar p$")
    axes[1].set_xlabel(r"$x$")
    plotting.panel_label(axes[0], "(a) mean flow rate of each cell layer")
    plotting.panel_label(axes[1], "(b) mean pressure of each cell layer")
    return {"plane": fig_plane, "sections": fig_sections, "profiles": fig_profiles}


if __name__ == "__main__":
    study.run("gyroid/single_run", compute, plot)
    show_fields(VTK_PATH, show="--show" in sys.argv)
