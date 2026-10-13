"""Mesh convergence of drag and lift for combinations of penalty parameters.

Nitsche penalty gamma_N against the ghost penalty gamma_g = gamma_u = gamma_p
(velocity and pressure ghost penalties kept equal). For every combination,
c_D and c_L are computed on the meshes of ``N_Y`` and compared with the
reference values of Nabh.
"""

import dataclasses
import itertools

import numpy as np

from cutfem import navier_stokes as ns
from cutfem import plotting, study
from cutfem.problems import dfg2d1

N_Y = [16, 24, 32, 48, 64, 96, 128]
GAMMA_N = [10.0, 30.0, 100.0]
GAMMA_G = [0.01, 0.1, 1.0]
PARAMS = dfg2d1.Params()
REFERENCE_ORDER = 4     # 2k for the forces (observed for c_D, not proven)


def refinement(params):
    """Solve on each mesh of ``N_Y``; returns one array per column."""
    rows, table = [], None
    for n_y in N_Y:
        with dfg2d1.build(dataclasses.replace(params, n_y=n_y)) as system:
            row = {"n_y": n_y, "h": system.h, "n_dofs": system.n_dofs}
            try:
                solution = dfg2d1.solve(system)
                row.update(dfg2d1.benchmark(system, solution))
                row.update(dfg2d1.errors(system, solution))
                row["newton"] = len(system.history) - 1
                row["Re_T"] = ns.cell_reynolds(system, solution[0])
            except RuntimeError as error:
                study.print0(f"  n_y = {n_y}: {error}")
                row.update(dict.fromkeys(
                    [*dfg2d1.QUANTITIES, *dfg2d1.ERRORS, "newton", "Re_T"], np.nan))
        if table is None:
            table = study.Table(row)
            table.header()
        table.row(row)
        rows.append(row)
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def compute():
    """Refinement sweep for every (gamma_N, gamma_g) combination."""
    study.print0(PARAMS)
    runs = []
    for gamma_N, gamma_g in itertools.product(GAMMA_N, GAMMA_G):
        study.print0(f"\ngamma_N = {gamma_N:g}, gamma_u = gamma_p = {gamma_g:g}")
        params = dataclasses.replace(PARAMS, gamma_N=gamma_N,
                                     gamma_u=gamma_g, gamma_p=gamma_g)
        runs.append({"gamma_N": gamma_N, "gamma_g": gamma_g,
                     "columns": refinement(params)})
    return {"params": PARAMS, "reference": dfg2d1.REFERENCE, "runs": runs}


def plot(data):
    """Relative errors of c_D (left) and c_L (right) versus h.

    One row per ghost penalty gamma_u = gamma_p, one curve per gamma_N.
    Axes are shared per column so the rows can be compared directly.
    """
    fig, axes = plotting.figure(nrows=len(GAMMA_G), ncols=2, aspect=0.75,
                                sharex=True, sharey="col")
    labels = {"c_D": r"$|c_D - c_D^{\mathrm{ref}}| / |c_D^{\mathrm{ref}}|$",
              "c_L": r"$|c_L - c_L^{\mathrm{ref}}| / |c_L^{\mathrm{ref}}|$"}

    for i, gamma_g in enumerate(GAMMA_G):
        runs = [r for r in data["runs"] if r["gamma_g"] == gamma_g]
        for j, key in enumerate(dfg2d1.QUANTITIES):
            ax = axes[i, j]
            for k, run in enumerate(runs):
                columns = run["columns"]
                ax.loglog(columns["h"], columns[f"err_{key}"],
                          color=plotting.CATEGORICAL[k], marker=plotting.MARKERS[k],
                          label=rf"$\gamma_N = {run['gamma_N']:g}$")
            reference_line(ax, runs, key)
            plotting.panel_label(ax, rf"$\gamma_u = \gamma_p = {gamma_g:g}$")
            ax.set_ylabel(labels[key])
    for ax in axes[-1]:
        ax.set_xlabel(r"$h$")

    plotting.legend(axes[0, 0], ncols=len(GAMMA_N) + 1)
    return fig


def reference_line(ax, runs, key):
    """Dashed C h^REFERENCE_ORDER over the whole h range, just below all curves."""
    h = np.asarray(runs[0]["columns"]["h"])
    ratios = [np.asarray(r["columns"][f"err_{key}"]) / h ** REFERENCE_ORDER
              for r in runs]
    C = 0.5 * np.nanmin(ratios)              # factor 2 below the lowest point
    ax.loglog(h, C * h ** REFERENCE_ORDER, color=plotting.INK, linestyle="--",
              linewidth=0.8, label=rf"$\mathcal{{O}}(h^{REFERENCE_ORDER})$")


if __name__ == "__main__":
    study.run("dfg2d1/parameter_convergence", compute, plot)
