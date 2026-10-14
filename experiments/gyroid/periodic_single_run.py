"""Flow through a periodic gyroid unit cell, driven by a body force (single run).

    (u . grad) u + grad p - nu Delta u = G e_x,   div u = 0   in Omega,
    u = 0 on the gyroid surface Gamma (Nitsche),
    u and p periodic on the six faces of the box [0, L]^3,

with Omega = {phi <= 0}, phi the Schoen gyroid of period L (one labyrinth,
porosity 1/2). The body force stands for a mean pressure gradient -G e_x
(p itself is periodic): this is the representative cell of the infinite
gyroid, without inlet, outlet or side walls. Reported: Newton history,
mean velocity <u> over the box (superficial velocity), apparent permeability
k = nu <u_x> / G, Re = <u_x> L / nu, and two conservation checks (with
periodic faces, int_Omega div u = int_Gamma u.n, both zero exactly).

Writes ``results/gyroid/periodic_single_run.pvd`` (velocity, pressure on a
fitted reparametrization of Omega) and the figures

* ``periodic_single_run_newton.pdf``: residual norms;
* ``periodic_single_run_profiles.pdf``: flow rate through each cross-section
  (constant for a divergence-free periodic flow) and mean pressure of each
  cell layer along x, from exact cut integrals;
* ``periodic_single_run_plane.pdf``: |u| and p on the plane z = L/2;
* ``periodic_single_run_sections.pdf``: u_x on four cross-sections;
* ``periodic_single_run.png``: |u| on the fluid domain (PyVista, 3D).

    python experiments/gyroid/periodic_single_run.py
    python experiments/gyroid/periodic_single_run.py --plot-only
"""

import resource
import sys
import time

import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import qugar.impl
import ufl
from mpi4py import MPI
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from cutfem import core, study
from cutfem import navier_stokes as ns

L = 1.0                     # unit-cell size (= box size)
N_CELL = 12                 # background cells per direction
G = 1.0                     # body force G e_x (mean pressure gradient)
PARAMS = ns.Params(nu=0.1, U=0.1, gamma_u=0.01, gamma_p=0.01)
DIAGNOSTIC_DEGREE = 8       # cut quadrature of the reported integrals
SECTIONS_X = (0.125, 0.375, 0.625, 0.875)   # cross-sections, in units of L
VTK_PATH = study.RESULTS_DIR / "gyroid" / "periodic_single_run.pvd"
FIGURE_PATH = study.FIGURES_DIR / "gyroid" / "periodic_single_run.png"


def peak_memory_gb():
    """Peak resident memory of this process (macOS: bytes, Linux: kB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def build():
    """Periodic gyroid cell; returns the Navier-Stokes system."""
    lo, hi = np.zeros(3), np.full(3, L)
    mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, qugar.impl.create_Schoen(periods=[1.0 / L] * 3), [N_CELL] * 3,
        lo, hi, exclude_empty_cells=True,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet, dtype=core.DTYPE)
    # Body force as a Function: no Constant next to Functions in a cut form
    f = dolfinx.fem.Function(core.lagrange_space(mesh, 1, shape=(3,)), name="f")
    f.interpolate(lambda x: np.vstack([np.full(x.shape[1], G),
                                       np.zeros((2, x.shape[1]))]))
    return ns.build(PARAMS, mesh, L / N_CELL, f=f, periodic=(lo, hi, (0, 1, 2)))


def quantities(s, uh):
    """Mean velocity, permeability, Reynolds numbers, conservation checks."""
    md = {"quadrature_degree": DIAGNOSTIC_DEGREE}
    m = core.cut_measures(s.mesh)
    dx, ds = m.dx(metadata=md), m.ds(metadata=md)
    one = dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0))
    mean = [core.integrate(uh[i] * dx) / L ** 3 for i in range(3)]
    flow_rate = mean[0] * L ** 2                    # through any cross-section
    return {"u_mean_x": mean[0], "u_mean_y": mean[1], "u_mean_z": mean[2],
            "porosity": core.integrate(one * dx) / L ** 3,
            "permeability": PARAMS.nu * mean[0] / G,
            "k_over_L2": PARAMS.nu * mean[0] / G / L ** 2,
            "Re": mean[0] * L / PARAMS.nu,
            "Re_cell": ns.cell_reynolds(s, uh),
            "wall_flux": core.integrate(ufl.dot(uh, m.n) * ds) / flow_rate,
            "divergence": core.integrate(ufl.div(uh) * dx) / flow_rate}


def layer_profiles(s, uh, ph):
    """Flow rate, mean pressure and fluid fraction of each cell layer along x.

    For the slab x_k < x < x_k + h, int u_x dV / h is the mean flow rate
    through its cross-sections, int p dV / |slab cap Omega| the mean
    pressure, |slab cap Omega| / (h L^2) the fluid fraction; exact cut
    integrals, assembled with a DG0 test function and summed per layer.
    """
    mesh, h = s.mesh, s.h
    W = dolfinx.fem.functionspace(mesh, basix.ufl.element("DG", mesh.basix_cell(), 0))
    w = ufl.TestFunction(W)
    dx = core.cut_measures(mesh).dx(metadata={"quadrature_degree": DIAGNOSTIC_DEGREE})
    one = dolfinx.fem.Constant(mesh, core.DTYPE(1.0))
    n_owned = mesh.topology.index_map(3).size_local
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(n_owned, dtype=np.int32))
    layer = np.floor(midpoints[:, 0] / h).astype(int)
    totals = []
    for integrand in (uh[0] * w, ph * w, one * w):
        vector = core.assemble_vector(integrand * dx)
        totals.append(np.bincount(layer, weights=vector.array[:n_owned],
                                  minlength=N_CELL))
        vector.destroy()
    flux, pressure, volume = totals
    return {"x": (np.arange(N_CELL) + 0.5) * h, "Q": flux / h,
            "p_mean": pressure / volume, "fraction": volume / (h * L ** 2)}


def compute():
    """Build, solve, report and write the fields."""
    study.print0(f"L = {L}, n_cell = {N_CELL}, G = {G}\n{PARAMS}")
    start = time.perf_counter()
    s = build()
    t_build = time.perf_counter() - start
    n_dofs, n_periodic = s.n_dofs, s.P.getSize()[1]
    study.print0(f"h = {s.h:.4f}, unknowns {n_dofs} ({n_periodic} periodic), "
                 f"build {t_build:.1f} s")
    with s:
        start = time.perf_counter()
        uh, ph = ns.solve(s, verbose=True)
        t_solve = time.perf_counter() - start
        values = quantities(s, uh)
        profiles = layer_profiles(s, uh, ph)
        history = list(s.history)
        core.save_vtk(VTK_PATH, s.mesh, {"uh": uh, "ph": ph}, degree=2)

    study.print0(f"solve {t_solve:.1f} s ({len(history) - 1} Newton steps), "
                 f"peak memory {peak_memory_gb():.2f} GB")
    for key, value in values.items():
        study.print0(f"  {key:>13} = {value: .6e}")
    deviation = np.max(np.abs(profiles["fraction"] - values["porosity"]))
    study.print0(f"  fluid fraction of every cell layer = porosity +- {deviation:.1e}")
    study.print0(f"Saved {VTK_PATH.relative_to(study.ROOT)}")
    return {"L": L, "n_cell": N_CELL, "G": G, "params": PARAMS, "n_dofs": n_dofs,
            "n_periodic": n_periodic, "values": values, "history": history,
            "profiles": profiles, "t_build": t_build, "t_solve": t_solve,
            "peak_memory_gb": peak_memory_gb()}


# Fields from the ParaView file


def read_fields(path):
    """Velocity, |u|, u_x and pressure on the fluid mesh, tessellated.

    VTK slices curved (Lagrange) cells inaccurately; tessellating into
    linear cells first makes the slices exact.
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


def off_grid(value):
    """Nearest background cell centre: a plane there never lies on cell faces."""
    h = L / N_CELL
    return (np.floor(value / h) + 0.5) * h


def sample_plane(grid, normal, offset, resolution):
    """Fields on a regular grid of the plane ``normal = offset``, solid masked.

    Returns ``(extent, {name: masked 2D array})`` in the in-plane coordinates.
    """
    import pyvista as pv

    in_plane = [i for i in range(3) if i != "xyz".index(normal)]
    t = (np.arange(resolution) + 0.5) / resolution * L
    A, B = np.meshgrid(t, t, indexing="xy")
    points = np.zeros((A.size, 3))
    points[:, in_plane[0]], points[:, in_plane[1]] = A.ravel(), B.ravel()
    points[:, "xyz".index(normal)] = offset
    probe = pv.PolyData(points).sample(grid)
    outside = probe.point_data["vtkValidPointMask"].reshape(A.shape) == 0
    fields = {key: np.ma.masked_where(outside, probe.point_data[key].reshape(A.shape))
              for key in ("|u|", "ph", "u_x")}
    return (0.0, L, 0.0, L), fields


def show_fields(path, show=False):
    """|u| on the boundary of the fluid domain (PyVista, 3D view)."""
    import pyvista as pv

    grid = read_fields(path)
    surface = grid.extract_surface(algorithm="dataset_surface")
    plotter = pv.Plotter(off_screen=not show, window_size=(1100, 900))
    plotter.add_text("velocity magnitude, body force along +x (periodic unit cell)",
                     font_size=10)
    plotter.add_mesh(surface, scalars="|u|", cmap="viridis",
                     clim=[0.0, float(grid.point_data["|u|"].max())],
                     scalar_bar_args=dict(title="", vertical=False, width=0.5,
                                          position_x=0.25, position_y=0.05, fmt="%.3f"))
    plotter.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, L)), style="wireframe",
                     color="black", line_width=1)
    plotter.view_isometric()
    plotter.show_axes()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(FIGURE_PATH), auto_close=True)
    study.print0(f"Saved {FIGURE_PATH.relative_to(study.ROOT)}")


def plot(data):
    """Newton history, layer profiles, mid-plane and cross-sections."""
    import matplotlib

    from cutfem import plotting

    def colormap(name):
        """Colour map with the solid (masked) region in light grey."""
        cmap = matplotlib.colormaps[name].copy()
        cmap.set_bad("0.85")
        return cmap

    # Newton residuals
    fig_newton, ax = plotting.figure(width=0.6)
    history = np.asarray(data["history"])
    ax.semilogy(np.arange(len(history)), history, marker="o")
    ax.set_xlabel(r"Newton iteration $k$")
    ax.set_ylabel(r"$\|P^T R_k\|$")
    ax.set_xticks(np.arange(len(history)))

    # Layer profiles along x
    profiles, values = data["profiles"], data["values"]
    flow_rate = values["u_mean_x"] * L ** 2
    fig_profiles, axes = plotting.figure(nrows=2, ncols=1, width=0.8, aspect=0.35,
                                         sharex=True)
    x = profiles["x"]
    axes[0].plot(x, np.asarray(profiles["Q"]) / flow_rate - 1.0, marker="o",
                 markersize=3)
    axes[0].axhline(0.0, color=plotting.INK, linestyle="--", linewidth=0.8)
    axes[0].set_ylabel(r"$\bar Q / Q - 1$")
    axes[1].plot(x, profiles["p_mean"], marker="o", markersize=3)
    axes[1].set_ylabel(r"$\bar p$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_xlim(0.0, L)
    plotting.panel_label(axes[0], "(a) flow rate through each cell layer")
    plotting.panel_label(axes[1], "(b) mean pressure of each cell layer")

    # Slices from the ParaView file
    grid = read_fields(VTK_PATH)
    u_max = float(grid.point_data["|u|"].max())
    p_abs = float(np.abs(grid.point_data["ph"]).max())
    image = dict(origin="lower", interpolation="nearest", rasterized=True)

    z_mid = off_grid(0.5 * L)
    extent, fields = sample_plane(grid, "z", z_mid, resolution=300)
    fig_plane, axes = plotting.figure(ncols=2, aspect=1.0, sharey=True)
    for ax, key, cmap, clim, label in (
            (axes[0], "|u|", "viridis", (0.0, u_max), r"$|u_h|$"),
            (axes[1], "ph", "coolwarm", (-p_abs, p_abs), r"$p_h$")):
        shown = ax.imshow(fields[key], extent=extent, cmap=colormap(cmap),
                          vmin=clim[0], vmax=clim[1], **image)
        fig_plane.colorbar(shown, ax=ax, label=label, orientation="horizontal",
                           pad=0.15)
        ax.set_xlabel(r"$x$")
        ax.grid(False)
    axes[0].set_ylabel(r"$y$")
    plotting.panel_label(axes[0], rf"(a) velocity magnitude, $z = {z_mid:.4g}$")
    plotting.panel_label(axes[1], rf"(b) pressure, $z = {z_mid:.4g}$")

    fig_sections, axes = plotting.figure(nrows=2, ncols=2, aspect=1.0,
                                         sharex=True, sharey=True)
    for k, (ax, xs) in enumerate(zip(axes.flat, SECTIONS_X, strict=True)):
        x_section = off_grid(xs * L)
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

    return {"newton": fig_newton, "profiles": fig_profiles, "plane": fig_plane,
            "sections": fig_sections}


if __name__ == "__main__":
    study.run("gyroid/periodic_single_run", compute, plot)
    show_fields(VTK_PATH, show="--show" in sys.argv)
