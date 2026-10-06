"""Cut position study for the Poisson CutFEM solver.

The disk is shifted by eps in [0, 0.4 h] on a fixed mesh. Without ghost
penalty, small cuts make the matrix nearly singular; the ghost penalty keeps
the smallest eigenvalue and the condition number bounded.
"""

from dataclasses import replace

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import poisson

PARAMS = poisson.Params(n_cells=32, beta=10.0)
TAU_VALUES = [0.0, 0.1, 1.0, 10.0]
H = (PARAMS.xmax[0] - PARAMS.xmin[0]) / PARAMS.n_cells
EPS = np.linspace(0.0, 0.4 * H, 20)
CENTERS = [(float(eps), 0.0) for eps in EPS]


def spectrum(s):
    """Return the smallest eigenvalue and the condition number of A."""
    return {"min_eig": core.eigenvalues(s.A)[0],
            "cond": core.condition_number(s.A)}


def compute():
    """Sweep the disk center for each tau; keep the worst cut geometry."""
    runs = []
    for tau in TAU_VALUES:
        study.print0(f"\ntau = {tau:g}")
        columns = study.sweep(poisson, replace(PARAMS, tau=tau), "center",
                              CENTERS, extras=spectrum)
        runs.append({"tau": tau, "columns": columns})

    # Worst conditioned cut without ghost penalty
    cond = runs[0]["columns"]["cond"]
    worst = int(np.argmax(cond))
    center = CENTERS[worst]
    study.print0(f"\nWorst cut (tau = 0): eps/h = {EPS[worst] / H:.3f}, "
                 f"cond = {cond[worst]:.4e}")
    mesh, _ = core.disk_mesh(PARAMS.n_cells, PARAMS.xmin, PARAMS.xmax,
                             center, PARAMS.radius)
    return {"eps_h": EPS / H, "runs": runs,
            "worst": {"center": center, "geometry": core.mesh_geometry(mesh)}}


def plot(data):
    """Sweep figure and worst-cut mesh figure."""
    fig, axes = plotting.figure(ncols=3, aspect=1.0)
    colors = plotting.ordered_colors([r["tau"] for r in data["runs"]])
    panels = [("min_eig", r"$\lambda_{\min}(A)$", "linear"),
              ("cond", r"$\kappa(A)$", "log"),
              ("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$", "log")]

    for ax, (key, ylabel, scale) in zip(axes, panels, strict=True):
        for i, (r, color) in enumerate(zip(data["runs"], colors, strict=True)):
            ax.plot(data["eps_h"], r["columns"][key], color=color,
                    marker=plotting.MARKERS[i], label=rf"$\tau = {r['tau']:g}$")
        ax.set_yscale(scale)
        ax.set_xlabel(r"$\varepsilon / h$")
        ax.set_ylabel(ylabel)
    axes[0].axhline(0.0, color=plotting.INK, linewidth=0.6)
    for r, color in zip(data["runs"], colors, strict=True):
        plotting.mark_nonfinite(axes[1], data["eps_h"], r["columns"]["cond"], color)
    plotting.legend_above(fig)

    worst = data["worst"]
    fig_mesh, ax = plotting.figure(width=0.6, aspect=1.0)
    handles = plotting.cut_mesh(ax, worst["geometry"], worst["center"],
                                PARAMS.radius, margin=2 * H)
    plotting.legend_above(fig_mesh, handles=handles, ncols=2)
    return {"sweep": fig, "worst_cut": fig_mesh}


if __name__ == "__main__":
    study.run("poisson/center_sweep", compute, plot)
