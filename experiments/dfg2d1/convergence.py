"""Mesh convergence of the Navier-Stokes CutFEM solver, DFG benchmark 2D-1.

The mesh has n_y cells across the channel (h = 0.41 / n_y, see
dfg2d1.channel_mesh). The default levels refine by about sqrt(2): twice as
many points as halving over the same range of h, and consecutive meshes are
not nested, so every level has its own, generic cut configuration.

For every level:

* c_D and c_L versus h, compared with the reference values (the
  relative errors and observed orders are printed and saved);
* |u_h| along the horizontal line y = y_c (through the cylinder and the
  wake) and the vertical line x = x_c (across the channel). There is no
  exact solution: convergence shows as profiles that stop changing,
  measured by the largest change between consecutive levels.

Usage::

    python convergence.py                         # default levels
    python convergence.py --n-y 16 32 64          # custom levels
    mpirun -n 4 python convergence.py --n-y 96 136
"""

import time

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import dfg2d1

BASE = dfg2d1.Params()
X_LINE = np.linspace(BASE.xmin[0], BASE.xmax[0], 2201)
Y_LINE = np.linspace(BASE.xmin[1], BASE.xmax[1], 411)
LABELS = {"c_D": r"$c_D$", "c_L": r"$c_L$"}
N_Y = [12, 17, 24, 34, 48, 68, 96, 136]     # ratio ~sqrt(2)
N_Y_FIT = 34    # resolved levels: n_y >= 34 (about 8 cells across the cylinder)
# Reference slope (no proven order for these functionals): 2k for the forces.
REFERENCE_SLOPES = {"c_D": 4, "c_L": 4}


def arguments(parser):
    """Refinement levels from the command line."""
    parser.add_argument("--n-y", type=int, nargs="+", default=N_Y,
                        help="cells across the channel (h = 0.41 / n_y)")


def speed_along(uh, points):
    """|u_h| at the points; NaN inside the cylinder."""
    speed = np.linalg.norm(core.eval_at_points(uh, points), axis=1)
    inside = np.hypot(points[:, 0] - BASE.center[0],
                      points[:, 1] - BASE.center[1]) < BASE.radius
    speed[inside] = np.nan
    return speed


def compute(args):
    """Solve every level; collect the benchmark quantities and the profiles."""
    xc, yc = BASE.center
    horizontal_points = np.column_stack([X_LINE, np.full_like(X_LINE, yc)])
    vertical_points = np.column_stack([np.full_like(Y_LINE, xc), Y_LINE])

    table = study.Table(["n_y", "h", "n_dofs", "picard", "time",
                         *(f"err_{q}" for q in dfg2d1.QUANTITIES),
                         "change x-line", "change y-line"], width=13)
    table.header()
    rows, horizontal, vertical = [], [], []
    for n_y in sorted(args.n_y):
        start = time.perf_counter()
        with dfg2d1.build(dfg2d1.Params(n_y=n_y)) as system:
            solution = dfg2d1.solve(system)
            row = {"n_y": n_y, "h": system.h, "n_dofs": system.n_dofs,
                   "picard": len(solution[2]),
                   **dfg2d1.benchmark(system, solution),
                   **dfg2d1.errors(system, solution)}
            horizontal.append(speed_along(solution[0], horizontal_points))
            vertical.append(speed_along(solution[0], vertical_points))
        row["time"] = time.perf_counter() - start
        changes = [np.nanmax(np.abs(line[-1] - line[-2])) if len(line) > 1 else np.nan
                   for line in (horizontal, vertical)]
        table.row([row[k] for k in ("n_y", "h", "n_dofs", "picard", "time")]
                  + [row[f"err_{q}"] for q in dfg2d1.QUANTITIES] + changes)
        rows.append(row)

    columns = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    for q in dfg2d1.QUANTITIES:
        orders = study.rates(columns["h"], columns[f"err_{q}"])
        study.print0(f"  observed orders {q:>4}: "
                     + "  ".join(f"{o:6.2f}" for o in orders))
    return {**columns, "x": X_LINE, "y": Y_LINE,
            "horizontal": horizontal, "vertical": vertical}


def set_h_ticks(ax):
    """Decimal h ticks: shorter than powers of ten on narrow panels."""
    ticks = [0.005, 0.01, 0.02]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel(r"$h$")


def plot_benchmark(data):
    """Plot the computed values versus h against the reference values."""
    fig, axes = plotting.figure(ncols=2, aspect=0.9)
    for ax, q in zip(axes, dfg2d1.QUANTITIES, strict=True):
        ax.semilogx(data["h"], data[q], marker=plotting.MARKERS[0], label="CutFEM")
        ax.axhline(dfg2d1.DFG_2D1_REF[q], color=plotting.INK, linestyle="--",
                   linewidth=0.8, label="reference")
        set_h_ticks(ax)
        ax.set_ylabel(LABELS[q])
    plotting.legend_above(fig)
    return fig


def plot_errors(data):
    """Relative errors versus h, with reference slopes on the resolved levels."""
    fig, axes = plotting.figure(ncols=2, aspect=0.9)
    h = np.asarray(data["h"])
    fine = np.asarray(data["n_y"]) >= N_Y_FIT
    for ax, q in zip(axes, dfg2d1.QUANTITIES, strict=True):
        err = np.asarray(data[f"err_{q}"])
        ax.loglog(h, err, marker=plotting.MARKERS[0])
        plotting.slope_triangle(ax, h[fine], err[fine], REFERENCE_SLOPES[q])
        set_h_ticks(ax)
        ax.set_ylabel(rf"relative error of {LABELS[q]}")
    return fig


def plot_profiles(data):
    """|u_h| along both lines, one curve per refinement level."""
    fig, (ax_x, ax_y) = plotting.figure(nrows=2, aspect=0.38)
    colors = plotting.ordered_colors(data["n_y"])
    xc, yc = BASE.center
    r = BASE.radius

    for color, h, line_x, line_y in zip(colors, data["h"], data["horizontal"],
                                        data["vertical"], strict=True):
        ax_x.plot(data["x"], line_x, color=color, label=rf"$h = {h:.3g}$")
        ax_y.plot(data["y"], line_y, color=color)

    ax_x.axvspan(xc - r, xc + r, color="0.9", zorder=0)
    ax_y.axvspan(yc - r, yc + r, color="0.9", zorder=0)
    ax_x.set_xlim(data["x"][0], data["x"][-1])
    ax_y.set_xlim(data["y"][0], data["y"][-1])
    plotting.panel_label(ax_x, rf"horizontal line $y = {yc:g}$")
    plotting.panel_label(ax_y, rf"vertical line $x = {xc:g}$")
    ax_x.set_xlabel(r"$x$")
    ax_y.set_xlabel(r"$y$")
    for ax in (ax_x, ax_y):
        ax.set_ylabel(r"$|u_h|$")
    plotting.legend_above(fig, ncols=4)
    return fig


def plot(data):
    """Values (convergence), errors (_errors) and profiles (_profiles)."""
    return {"": plot_benchmark(data), "errors": plot_errors(data),
            "profiles": plot_profiles(data)}


if __name__ == "__main__":
    study.run("dfg2d1/convergence", compute, plot, arguments=arguments)
