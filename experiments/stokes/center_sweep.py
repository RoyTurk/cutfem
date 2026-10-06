"""Cut position study for the Stokes CutFEM solver.

The disk center moves along the x-axis on a fixed mesh; for each ghost
penalty parameter, the conditioning of the velocity block and the errors
are recorded.
"""

from dataclasses import replace

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import stokes

PARAMS = stokes.Params(n_cells=32, xmin=(-1.243, -1.243), xmax=(1.243, 1.243))
GAMMA_G_VALUES = [0.0, 0.1, 1.0]
CENTER_X = np.linspace(-0.3, 0.3, 41)
PANELS = [("cond", r"$\kappa(A_{uu})$"),
          ("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$"),
          ("L2_p", r"$\|p - p_h\|_{L^2(\Omega)}$")]


def conditioning(s):
    """Return the condition number of the velocity block."""
    return {"cond": core.condition_number(s.A_uu)}


def compute():
    """Center sweep for each gamma_g."""
    centers = [(float(cx), 0.0) for cx in CENTER_X]
    results = []
    for gamma_g in GAMMA_G_VALUES:
        study.print0(f"\ngamma_g = {gamma_g:g}")
        columns = study.sweep(stokes, replace(PARAMS, gamma_g=gamma_g), "center",
                              centers, extras=conditioning)
        results.append({"gamma_g": gamma_g, "columns": columns})
    return {"center_x": CENTER_X, "runs": results}


def plot(data):
    """Condition number and errors versus the disk center position."""
    fig, axes = plotting.figure(ncols=3, aspect=1.0)
    colors = plotting.ordered_colors([r["gamma_g"] for r in data["runs"]])

    for ax, (key, ylabel) in zip(axes, PANELS, strict=True):
        for i, (r, color) in enumerate(zip(data["runs"], colors, strict=True)):
            ax.semilogy(data["center_x"], r["columns"][key], color=color,
                        marker=plotting.MARKERS[i], markevery=4,
                        label=rf"$\gamma_g = {r['gamma_g']:g}$")
        ax.set_xlabel(r"disk center $x_c$")
        ax.set_ylabel(ylabel)
    for r, color in zip(data["runs"], colors, strict=True):
        plotting.mark_nonfinite(axes[0], data["center_x"], r["columns"]["cond"], color)

    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("stokes/center_sweep", compute, plot)
