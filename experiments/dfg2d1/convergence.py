"""Mesh convergence of drag and lift for the DFG 2D-1 benchmark.

c_D and c_L (Nitsche-consistent force on the cylinder) versus h, and their
relative errors against the reference values of Nabh.
"""

import dataclasses

import numpy as np

from cutfem import navier_stokes as ns
from cutfem import plotting, study
from cutfem.problems import dfg2d1

N_Y = [16, 24, 32, 48, 64, 96, 128]
PARAMS = dfg2d1.Params()
PANELS = [("c_D", r"$c_D$"), ("c_L", r"$c_L$")]


def compute():
    """Solve on each mesh; record the coefficients, errors and Newton steps."""
    study.print0(PARAMS)
    rows, table = [], None
    for n_y in N_Y:
        params = dataclasses.replace(PARAMS, n_y=n_y)
        with dfg2d1.build(params) as system:
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

    columns = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    study.print0("  Observed orders:")
    for key in dfg2d1.ERRORS:
        observed = "  ".join(f"{r:6.2f}" for r in study.rates(columns["h"], columns[key]))
        study.print0(f"  {key:>8}: {observed}")
    return {"params": PARAMS, "reference": dfg2d1.REFERENCE, "columns": columns}


def plot(data):
    """Coefficients against the reference (top), relative errors (bottom)."""
    columns, reference = data["columns"], data["reference"]
    fig, axes = plotting.figure(nrows=2, ncols=2, aspect=0.75)
    h = columns["h"]

    for j, (key, label) in enumerate(PANELS):
        ax = axes[0, j]
        ax.semilogx(h, columns[key], marker=plotting.MARKERS[0], label="CutFEM")
        ax.axhline(reference[key], color=plotting.INK, linestyle="--",
                   linewidth=0.8, label="reference")
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(label)

        ax = axes[1, j]
        ax.loglog(h, columns[f"err_{key}"], marker=plotting.MARKERS[0])
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(rf"$|{label[1:-1]} - {label[1:-1]}^{{\mathrm{{ref}}}}|"
                      rf" / |{label[1:-1]}^{{\mathrm{{ref}}}}|$")

    plotting.legend(axes[0, 0])
    return fig


if __name__ == "__main__":
    study.run("dfg2d1/convergence", compute, plot)
