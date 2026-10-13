"""Sensitivity of DFG 2D-1 to how the background mesh cuts the cylinder.

The geometry and the reference values stay fixed: n_y is fixed and n_x
varies in a narrow band around the square-cell value, so the cells are
slightly non-square (aspect ratio ~0.97-1.03), the resolution is
practically the same, and only the cut of the cylinder changes. For each
mesh and ghost penalty: drag and lift errors, kappa(A) and the Newton
outcome (a stalled solve keeps its last iterate), plus the smallest fluid
fraction |T cap Omega| / |T| of a cut cell, the measure of a bad cut.
"""

import dataclasses
import sys
from pathlib import Path

import basix.ufl
import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI

from cutfem import core, plotting, study
from cutfem import navier_stokes as ns
from cutfem.problems import dfg2d1

sys.path.insert(0, str(Path(__file__).parent))
from summary import broken_axis, mark_stalled  # noqa: E402

N_Y = 48
N_X = list(range(250, 271))         # square cells: n_x = 258
GAMMA_N = 30.0
GAMMA_G = [0.0, 0.01, 0.1]


def smallest_cut_fraction(s, params):
    """min over cut cells of |T cap Omega| / |T| (DG0 test function: cell volumes)."""
    W = dolfinx.fem.functionspace(
        s.mesh, basix.ufl.element("DG", s.mesh.basix_cell(), 0))
    volumes = core.assemble_vector(ufl.TestFunction(W) * s.measures.dx)
    cell_area = (params.length / params.n_x) * (params.height / params.n_y)
    cut = s.mesh.get_cut_cells()
    local = float(np.min(volumes.array[cut]) / cell_area) if len(cut) else 1.0
    volumes.destroy()
    return s.mesh.comm.allreduce(local, op=MPI.MIN)


def run(params):
    """One solve; keeps the last iterate if Newton stalls."""
    with dfg2d1.build(params) as system:
        row = {"n_x": params.n_x, "h": system.h,
               "fraction": smallest_cut_fraction(system, params), "stalled": 0}
        try:
            solution = dfg2d1.solve(system)
            row["newton"] = len(system.history) - 1
        except RuntimeError as error:
            study.print0(f"  n_x = {params.n_x}: {error}; keeping the last iterate")
            solution = (system.uh, system.ph)
            ns.assemble_jacobian(system)
            row["stalled"], row["newton"] = 1, params.max_iter
        row.update(dfg2d1.errors(system, solution))
        row["kappa"] = ns.condition_number(system)[0]
    return row


def compute():
    """Every n_x for every ghost penalty."""
    base = dataclasses.replace(dfg2d1.Params(), n_y=N_Y, gamma_N=GAMMA_N)
    study.print0(base)
    runs = []
    for gamma_g in GAMMA_G:
        study.print0(f"\ngamma_u = gamma_p = {gamma_g:g}")
        rows, table = [], None
        for n_x in N_X:
            row = run(dataclasses.replace(base, n_x=n_x, gamma_u=gamma_g,
                                          gamma_p=gamma_g))
            if table is None:
                table = study.Table(row)
                table.header()
            table.row(row)
            rows.append(row)
        runs.append({"gamma_g": gamma_g,
                     "columns": {k: np.array([r[k] for r in rows]) for k in rows[0]}})
    return {"n_y": N_Y, "gamma_N": GAMMA_N, "runs": runs}


def plot(data):
    """Errors, kappa and the smallest cut fraction versus n_x."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    fig = plt.figure(figsize=(plotting.TEXTWIDTH, 0.82 * plotting.TEXTWIDTH))
    grid = fig.add_gridspec(2, 2)
    ax_drag = fig.add_subplot(grid[0, 0])
    ax_lift = fig.add_subplot(grid[0, 1], sharex=ax_drag)
    cell = grid[1, 0].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.0)
    ax_k_top = fig.add_subplot(cell[0], sharex=ax_drag)
    ax_k = fig.add_subplot(cell[1], sharex=ax_drag)
    ax_cut = fig.add_subplot(grid[1, 1], sharex=ax_drag)
    style = {g: (plotting.CATEGORICAL[i], plotting.MARKERS[i])
             for i, g in enumerate(GAMMA_G)}

    for run in data["runs"]:
        color, marker = style[run["gamma_g"]]
        c = dict(run["columns"])
        for ax, key in ((ax_drag, "err_c_D"), (ax_lift, "err_c_L")):
            ax.semilogy(c["n_x"], c[key], color=color, marker=marker)
            mark_stalled(ax, c, key, color, x="n_x")
        ax = ax_k_top if run["gamma_g"] == 0.0 else ax_k
        ax.semilogy(c["n_x"], c["kappa"], color=color, marker=marker)
        mark_stalled(ax, c, "kappa", color, x="n_x")

    fraction = data["runs"][0]["columns"]
    ax_cut.semilogy(fraction["n_x"], fraction["fraction"], color=plotting.INK,
                    marker="o", markersize=3)

    broken_axis(ax_k_top, ax_k)
    ax_k_top.yaxis.set_major_locator(LogLocator(base=1e4))
    ax_drag.set_ylabel(r"$|c_D - c_D^{\mathrm{ref}}| / |c_D^{\mathrm{ref}}|$")
    ax_lift.set_ylabel(r"$|c_L - c_L^{\mathrm{ref}}| / |c_L^{\mathrm{ref}}|$")
    ax_k.set_ylabel(r"$\kappa(A)$")
    ax_k.yaxis.set_label_coords(-0.16, 0.75)
    ax_cut.set_ylabel(r"$\min_T |T \cap \Omega| / |T|$")
    plotting.panel_label(ax_drag, "(a) drag")
    plotting.panel_label(ax_lift, "(b) lift")
    plotting.panel_label(ax_k_top, "(c) condition number")
    plotting.panel_label(ax_cut, "(d) smallest cut-cell fraction")
    for ax in (ax_k, ax_cut):
        ax.set_xlabel(rf"$n_x$ ($n_y = {data['n_y']}$)")
    for ax in (ax_drag, ax_lift, ax_k_top):
        ax.tick_params(axis="x", which="both", labelbottom=False)

    handles = [Line2D([], [], color=style[g][0], marker=style[g][1],
                      label=rf"$\gamma_u = \gamma_p = {g:g}$") for g in GAMMA_G]
    handles.append(Line2D([], [], color=style[0.0][0], marker="o", markersize=8,
                          markerfacecolor="white", linestyle="none",
                          label="Newton stalled (last iterate)"))
    fig.legend(handles=handles, loc="outside lower center", ncols=2)
    return fig


if __name__ == "__main__":
    study.run("dfg2d1/cut_sensitivity", compute, plot)
