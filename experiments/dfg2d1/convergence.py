"""Mesh convergence of the Navier-Stokes CutFEM solver, DFG benchmark 2D-1.

c_D, c_L and dp versus h, compared with the reference values (the relative
errors are printed and saved in the data file).

Usage::

    python convergence.py                         # default levels
    python convergence.py --refs 0.25 0.5 1 2     # custom levels
    mpirun -n 4 python convergence.py --refs 1 2 4
"""

import time

import numpy as np

from cutfem import plotting, study
from cutfem.problems import dfg2d1

LABELS = {"c_D": r"$c_D$", "c_L": r"$c_L$", "dp": r"$\Delta p$"}


def arguments(parser):
    """Refinement levels from the command line."""
    parser.add_argument("--refs", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0],
                        help="refinement levels (h ~ 0.01 / ref)")


def compute(args):
    """Solve every refinement level and collect the benchmark quantities."""
    table = study.Table(["ref", "h", "n_dofs", "picard", "time",
                         *(f"err_{q}" for q in dfg2d1.QUANTITIES)])
    table.header()
    rows = []
    for ref in sorted(args.refs):
        start = time.perf_counter()
        with dfg2d1.build(dfg2d1.Params(ref=ref)) as system:
            solution = dfg2d1.solve(system)
            row = {"ref": ref, "h": system.h, "n_dofs": system.n_dofs,
                   "picard": len(solution[2]),
                   **dfg2d1.benchmark(system, solution),
                   **dfg2d1.errors(system, solution)}
        row["time"] = time.perf_counter() - start
        table.row(row)
        rows.append(row)

    columns = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    for q in dfg2d1.QUANTITIES:
        orders = study.rates(columns["h"], columns[f"err_{q}"])
        study.print0(f"  observed orders {q:>4}: "
                     + "  ".join(f"{o:6.2f}" for o in orders))
    return columns


def plot(c):
    """Plot the computed values versus h against the reference values."""
    fig, axes = plotting.figure(ncols=3, aspect=0.9)
    h = c["h"]
    for ax, q in zip(axes, dfg2d1.QUANTITIES, strict=True):
        ax.semilogx(h, c[q], marker="o", label="CutFEM")
        ax.axhline(dfg2d1.DFG_2D1_REF[q], color=plotting.INK, linestyle="--",
                   linewidth=0.8, label="reference")
        plotting.panel_label(ax, LABELS[q])
        ax.set_xticks(h, [f"{x:.2g}" for x in h])
        ax.minorticks_off()
        ax.set_xlabel(r"$h$")
    axes[0].set_ylabel("value")
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("dfg2d1/convergence", compute, plot, arguments=arguments)
