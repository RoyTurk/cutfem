"""Condition number of the DFG 2D-1 Jacobian versus h.

kappa(A) of the Newton Jacobian at the solution (ghost penalty included),
on the free unknowns (strong Dirichlet rows and columns removed), for
several ghost penalties gamma_u = gamma_p. With the ghost penalty, kappa
should grow like h^-2 independently of how the cylinder cuts the mesh.
"""

import dataclasses

import numpy as np

from cutfem import navier_stokes as ns
from cutfem import plotting, study
from cutfem.problems import dfg2d1

N_Y = [16, 24, 32, 48, 64, 96, 128]
GAMMA_G = [0.0, 0.01, 0.1, 1.0]
PARAMS = dfg2d1.Params()


def refinement(params):
    """Solve on each mesh, then estimate kappa of the Jacobian."""
    rows, table = [], None
    for n_y in N_Y:
        with dfg2d1.build(dataclasses.replace(params, n_y=n_y)) as system:
            row = {"n_y": n_y, "h": system.h, "n_dofs": system.n_dofs}
            try:
                dfg2d1.solve(system)
                row["newton"] = len(system.history) - 1
                row["kappa"], row["sigma_max"], row["sigma_min"] = \
                    ns.condition_number(system)
            except RuntimeError as error:
                study.print0(f"  n_y = {n_y}: {error}")
                row.update(dict.fromkeys(["newton", "kappa", "sigma_max",
                                          "sigma_min"], np.nan))
        if table is None:
            table = study.Table(row)
            table.header()
        table.row(row)
        rows.append(row)
    columns = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    observed = "  ".join(f"{-r:6.2f}" for r in study.rates(columns["h"], columns["kappa"]))
    study.print0(f"  observed growth orders of kappa (h^-p): {observed}")
    return columns


def compute():
    """Refinement sweep for every ghost penalty."""
    study.print0(PARAMS)
    runs = []
    for gamma_g in GAMMA_G:
        study.print0(f"\ngamma_u = gamma_p = {gamma_g:g}")
        params = dataclasses.replace(PARAMS, gamma_u=gamma_g, gamma_p=gamma_g)
        runs.append({"gamma_g": gamma_g, "columns": refinement(params)})
    return {"params": PARAMS, "runs": runs}


def plt_rc(key):
    import matplotlib.pyplot as plt
    return plt.rcParams[key]


def broken_axis(fig, ax_top, ax_bottom):
    """Join two stacked axes into one panel with a broken y-axis (// marks)."""
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.02)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle="none",
                 color=plotting.INK, mec=plotting.INK, mew=0.6, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **marks)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **marks)


def plot(data):
    """kappa(A) versus h on a broken y-axis.

    The unstabilized curve (gamma_u = gamma_p = 0) is many decades above
    the others: it is drawn above the break, the ghost penalty curves and
    the h^-2 reference below it.
    """
    fig, (ax_top, ax_bottom) = plotting.figure(
        nrows=2, width=0.8, aspect=0.32, sharex=True,
        gridspec_kw={"height_ratios": [1, 2]})
    broken_axis(fig, ax_top, ax_bottom)
    for i, run in enumerate(data["runs"]):
        columns = run["columns"]
        ax = ax_top if run["gamma_g"] == 0.0 else ax_bottom
        ax.loglog(columns["h"], columns["kappa"], color=plotting.CATEGORICAL[i],
                  marker=plotting.MARKERS[i],
                  label=rf"$\gamma_u = \gamma_p = {run['gamma_g']:g}$")
    stabilized = [r for r in data["runs"] if r["gamma_g"] > 0.0]
    lowest = min(stabilized, key=lambda r: r["columns"]["kappa"][-1])["columns"]
    h = np.asarray(lowest["h"])
    ax_bottom.loglog(h, lowest["kappa"][-1] * (h / h[-1]) ** -2, color=plotting.INK,
                     linestyle="--", linewidth=0.8, label=r"$\mathcal{O}(h^{-2})$")
    fig.supylabel(r"$\kappa(A)$", fontsize=plt_rc("axes.labelsize"))
    ax_bottom.set_xlabel(r"$h$")

    from matplotlib.ticker import LogLocator

    ax_top.yaxis.set_major_locator(LogLocator(base=1e3))
    handles = [line for ax in (ax_top, ax_bottom) for line in ax.get_lines()
               if not line.get_label().startswith("_")]          # no break marks
    plotting.legend(ax_bottom, ncols=3, handles=handles)
    return fig


if __name__ == "__main__":
    study.run("dfg2d1/condition", compute, plot)
