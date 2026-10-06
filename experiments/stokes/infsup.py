"""Numerical inf-sup test for the Stokes CutFEM (serial only).

Chapelle & Bathe (1993) test adapted to the stabilized Nitsche CutFEM of
Burman & Hansbo (2014); see :func:`cutfem.problems.stokes.inf_sup`.

Study 1: shift the disk center across one cell width at fixed h.
Study 2: mesh refinement, worst case over several shifts per h.
"""

from dataclasses import replace

import numpy as np

from cutfem import plotting, study
from cutfem.problems import stokes

BASE = stokes.Params(gamma_p=0.0, xmin=(-1.243, -1.243), xmax=(1.243, 1.243))
LENGTH = BASE.xmax[0] - BASE.xmin[0]
GAMMA_G_VALUES = [0.0, 0.01, 0.1, 1.0]

N_CELLS_SWEEP = 32
SHIFTS = np.linspace(0.0, 1.0, 41)          # center shift / h
N_CELLS_REFINE = [16, 24, 32, 48, 64]
N_SHIFTS_REFINE = 6


def inf_sup(n_cells, shift, gamma_g):
    """Inf-sup data for the disk shifted by ``shift * h`` along x."""
    h = LENGTH / n_cells
    params = replace(BASE, n_cells=n_cells, gamma_g=gamma_g,
                     center=(float(shift * h), 0.0))
    with stokes.build(params) as system:
        return stokes.inf_sup(system)


def compute():
    """Shift sweep at fixed h, then worst case over shifts versus h."""
    table = study.Table(["gamma_g", "s/h", "beta", "kappa", "n_small"])
    study.print0("\nStudy 1: center shift sweep")
    table.header()
    sweep = []
    for gamma_g in GAMMA_G_VALUES:
        rows = [inf_sup(N_CELLS_SWEEP, s, gamma_g) for s in SHIFTS]
        for s, r in zip(SHIFTS, rows, strict=True):
            table.row([gamma_g, s, r["beta"], r["kappa"], r["n_small"]])
        sweep.append({"gamma_g": gamma_g,
                      **{k: np.array([r[k] for r in rows]) for k in rows[0]}})

    table = study.Table(["gamma_g", "n_cells", "beta_min", "kappa_max"])
    study.print0("\nStudy 2: refinement, worst case over shifts")
    table.header()
    refine = []
    for gamma_g in GAMMA_G_VALUES:
        beta_min, kappa_max = [], []
        for n in N_CELLS_REFINE:
            rows = [inf_sup(n, k / N_SHIFTS_REFINE, gamma_g)
                    for k in range(N_SHIFTS_REFINE)]
            beta_min.append(np.nanmin([r["beta"] for r in rows]))
            kappa_max.append(np.nanmax([r["kappa"] for r in rows]))
            table.row([gamma_g, n, beta_min[-1], kappa_max[-1]])
        refine.append({"gamma_g": gamma_g, "beta_min": beta_min,
                       "kappa_max": kappa_max})

    h = LENGTH / np.array(N_CELLS_REFINE)
    return {"shifts": SHIFTS, "sweep": sweep, "h": h, "refine": refine}


def _panels(fig, axes, x, runs, keys, xlabel, log_x):
    colors = plotting.ordered_colors([r["gamma_g"] for r in runs])
    for ax, (key, ylabel) in zip(axes, keys, strict=True):
        for i, (r, color) in enumerate(zip(runs, colors, strict=True)):
            values = np.asarray(r[key], dtype=float)
            ok = np.isfinite(values) & (values > 0)
            ax.plot(x[ok], values[ok], color=color, marker=plotting.MARKERS[i],
                    markevery=None if log_x else 4,
                    label=rf"$\gamma_g = {r['gamma_g']:g}$")
        ax.set_yscale("log")
        ax.set_xscale("log" if log_x else "linear")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for r, color in zip(runs, colors, strict=True):
            plotting.mark_nonfinite(ax, x, r[key], color)
    plotting.legend_above(fig)


def plot(data):
    """Shift sweep figure and refinement figure (beta_h and kappa)."""
    fig_sweep, axes = plotting.figure(ncols=2, aspect=0.85)
    _panels(fig_sweep, axes, data["shifts"], data["sweep"],
            [("beta", r"$\beta_h$"),
             ("kappa", r"$\lambda_{\max} / \lambda_{\min}$")],
            r"center shift $s / h$", log_x=False)

    fig_refine, axes = plotting.figure(ncols=2, aspect=0.85)
    _panels(fig_refine, axes, data["h"], data["refine"],
            [("beta_min", r"$\min_s \beta_h$"),
             ("kappa_max", r"$\max_s \lambda_{\max} / \lambda_{\min}$")],
            r"$h$", log_x=True)
    return {"sweep": fig_sweep, "refinement": fig_refine}


if __name__ == "__main__":
    study.run("stokes/infsup", compute, plot)
