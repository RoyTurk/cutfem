"""Velocity magnitude profiles of the Navier-Stokes CutFEM solution (DFG 2D-1).

|u_h| along the horizontal line y = y_c (through the cylinder and the wake)
and the vertical line x = x_c (across the channel), for several refinement
levels. There is no exact solution: convergence shows as profiles that stop
changing, measured by the largest change between consecutive levels.

Usage::

    python profiles.py                         # default levels
    python profiles.py --refs 0.5 1 2 4        # custom levels
"""

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import navier_stokes as ns

BASE = ns.Params()
X_LINE = np.linspace(BASE.xmin[0], BASE.xmax[0], 2201)
Y_LINE = np.linspace(BASE.xmin[1], BASE.xmax[1], 411)


def arguments(parser):
    """Refinement levels from the command line."""
    parser.add_argument("--refs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="refinement levels (h ~ 0.01 / ref)")


def speed_along(uh, points):
    """|u_h| at the points; NaN inside the cylinder."""
    speed = np.linalg.norm(core.eval_at_points(uh, points), axis=1)
    inside = np.hypot(points[:, 0] - BASE.center[0],
                      points[:, 1] - BASE.center[1]) < BASE.radius
    speed[inside] = np.nan
    return speed


def compute(args):
    """Solve every level and sample |u_h| on both lines."""
    xc, yc = BASE.center
    horizontal_points = np.column_stack([X_LINE, np.full_like(X_LINE, yc)])
    vertical_points = np.column_stack([np.full_like(Y_LINE, xc), Y_LINE])

    table = study.Table(["ref", "h", "picard", "max |u_h|", "change x-line",
                         "change y-line"], width=14)
    table.header()
    refs, h, horizontal, vertical = sorted(args.refs), [], [], []
    for ref in refs:
        with ns.build(ns.Params(ref=ref)) as system:
            uh, _, history = ns.solve(system)
            h.append(system.h)
            horizontal.append(speed_along(uh, horizontal_points))
            vertical.append(speed_along(uh, vertical_points))
        changes = [np.nanmax(np.abs(line[-1] - line[-2])) if len(line) > 1 else np.nan
                   for line in (horizontal, vertical)]
        table.row([ref, h[-1], len(history), np.nanmax(horizontal[-1]), *changes])

    return {"refs": refs, "h": h, "x": X_LINE, "y": Y_LINE,
            "horizontal": horizontal, "vertical": vertical}


def plot(data):
    """|u_h| along both lines, one curve per refinement level."""
    fig, (ax_x, ax_y) = plotting.figure(nrows=2, aspect=0.38)
    colors = plotting.ordered_colors(data["refs"])
    xc, yc = BASE.center
    r = BASE.radius

    for color, h, line_x, line_y in zip(colors, data["h"], data["horizontal"],
                                        data["vertical"], strict=True):
        label = rf"$h = {h:.3g}$"
        ax_x.plot(data["x"], line_x, color=color, label=label)
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
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("navier_stokes/profiles", compute, plot, arguments=arguments)
