"""Stationary Navier-Stokes flow through the gyroid duct: a first run.

Solves the problem of cutfem.problems.gyroid_duct (Picard, starting from
Stokes), then reports and saves:

* the injected flux Q_in, the flux of each unit cell (exact quadrature)
  and the flux Q(x) on cross-sections (sampled), which must all agree;
* the mean pressure per cell and on cross-sections, the pressure drop per
  cell and the apparent (Darcy) permeability k = nu Q / dp;
* |u| on cross-sections x = const and on the lengthwise plane y = 1/2;
* velocity and pressure for ParaView (results/gyroid/*.pvd).

Usage::

    export CC=/usr/bin/clang                     # JIT linking, see README
    python duct_flow.py                          # n_per_cell = 4, Re = 1
    python duct_flow.py --n-per-cell 8 --re 10
    python duct_flow.py --plot-only
"""

import time

import numpy as np

from cutfem import plotting, study
from cutfem.problems import gyroid_duct

NAME = "gyroid/duct_flow"
SECTIONS = [0.5, 1.5, 2.5, 3.5]      # cross-sections shown (cell centres)


def arguments(parser):
    """Mesh and flow options from the command line."""
    defaults = gyroid_duct.Params()
    parser.add_argument("--n-x", type=int, default=defaults.n_x)
    parser.add_argument("--n-per-cell", type=int, default=4,
                        help="background cells per unit cell and direction")
    parser.add_argument("--c", type=float, default=defaults.c)
    parser.add_argument("--re", type=float, default=defaults.Re)
    parser.add_argument("--n-sample", type=int, default=60,
                        help="sample points per unit length on the slices")


def vtk_path(params):
    return study.RESULTS_DIR / "gyroid" / f"duct_flow_n{params.n_per_cell}_re{params.Re:g}.pvd"


def compute(args):
    """Solve, then collect quantities, profiles and slices."""
    params = gyroid_duct.Params(n_x=args.n_x, n_per_cell=args.n_per_cell,
                                c=args.c, Re=args.re)
    start = time.perf_counter()
    with gyroid_duct.build(params) as s:
        study.print0(params)
        study.print0(f"h = {s.h:.4f}, n_dofs = {s.n_dofs}, "
                     f"build {time.perf_counter() - start:.0f} s")
        solution = gyroid_duct.solve(s, verbose=True)
        study.print0(f"solved after {time.perf_counter() - start:.0f} s")

        q = gyroid_duct.quantities(s, solution)
        xs = np.linspace(0.0, params.n_x, 8 * params.n_x + 1)
        xs[0], xs[-1] = 1e-6, params.n_x - 1e-6     # stay inside the mesh
        flux_x, pressure_x = gyroid_duct.section_profiles(s, solution, xs, args.n_sample)
        slices = [gyroid_duct.velocity_slice(s, solution, 0, x0, args.n_sample)
                  for x0 in SECTIONS if x0 < params.n_x]
        lengthwise = gyroid_duct.velocity_slice(s, solution, 1, 0.5, args.n_sample)
        gyroid_duct.save_vtk(vtk_path(params), s, solution)
        n_dofs = s.n_dofs

    study.print0(f"\n  Q_in     {q['Q_in']:.6f}   (injected, fluid part of the inlet)")
    study.print0(f"  Q_mean   {q['Q_mean']:.6f}   (mean over the cells, rel. diff "
                 f"{abs(q['Q_mean'] / q['Q_in'] - 1):.2e})")
    study.print0("  Q_cells  " + "  ".join(f"{v:.4f}" for v in q["Q_cells"]))
    study.print0("  p_cells  " + "  ".join(f"{v:.4g}" for v in q["p_cells"]))
    study.print0(f"  dp_cell  {q['dp_cell']:.4g}   k_app {q['k_app']:.4e}   "
                 f"||div u||  {q['div_L2']:.3e}")
    study.print0(f"Saved {vtk_path(params).relative_to(study.ROOT)}")
    return {"params": params, "n_dofs": n_dofs, **q, "x": xs, "flux_x": flux_x,
            "pressure_x": pressure_x, "slices": slices, "lengthwise": lengthwise,
            "runtime": time.perf_counter() - start}


def _speed_image(ax, sl, vmax, extent):
    """|u| on a slice; the solid (NaN) in grey."""
    ax.set_facecolor("0.55")
    im = ax.imshow(np.asarray(sl["speed"], dtype=float).T, origin="lower", extent=extent, cmap="viridis",
                   vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")
    return im


def plot(data):
    """Velocity slices, and flux / pressure along the duct."""
    p = data["params"]
    slices, lw = data["slices"], data["lengthwise"]
    vmax = np.nanmax([np.nanmax(np.asarray(sl["speed"], dtype=float))
                      for sl in slices + [lw]])

    # Cross-sections x = const (looking downstream: y right, z up)
    ncols = 2
    nrows = -(-len(slices) // ncols)
    fig_s, axes = plotting.figure(nrows=nrows, ncols=ncols, aspect=0.9 * nrows / ncols,
                                  sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, sl in zip(axes, slices):
        im = _speed_image(ax, sl, vmax, (0, 1, 0, 1))
        plotting.panel_label(ax, f"$x = {sl['value']:g}$")
    for ax in axes[len(slices):]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("$y$")
    for ax in axes[::ncols]:
        ax.set_ylabel("$z$")
    fig_s.colorbar(im, ax=axes.tolist(), label=r"$|\mathbf{u}_h|$", shrink=0.9)

    # Lengthwise plane y = 1/2
    fig_l, ax = plotting.figure(aspect=0.32)
    im = _speed_image(ax, lw, vmax, (0, p["n_x"], 0, 1))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    fig_l.colorbar(im, ax=ax, label=r"$|\mathbf{u}_h|$", shrink=0.9)

    # Flux and mean pressure along the duct
    fig_q, (ax_q, ax_p) = plotting.figure(nrows=2, aspect=0.7, sharex=True)
    ax_q.plot(data["x"], data["flux_x"], marker="o", ms=3, label="$Q(x)$, sampled")
    centres = np.arange(p["n_x"]) + 0.5
    ax_q.plot(centres, data["Q_cells"], "s", label="cell mean, exact")
    ax_q.axhline(data["Q_in"], color=plotting.INK, lw=0.8, ls="--", label=r"$Q_{in}$")
    ax_q.set_ylabel("$Q$")
    ax_p.plot(data["x"], data["pressure_x"], marker="o", ms=3)
    ax_p.plot(centres, data["p_cells"], "s")
    ax_p.set_ylabel(r"$\bar p$")
    ax_p.set_xlabel("$x$")
    plotting.legend(ax_p, handles=ax_q.get_legend_handles_labels()[0])
    return {"sections": fig_s, "lengthwise": fig_l, "profiles": fig_q}


if __name__ == "__main__":
    study.run(NAME, compute, plot, arguments)
