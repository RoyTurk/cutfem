"""Numerical inf-sup test for the Stokes CutFEM (serial only).

Chapelle & Bathe (1993) test adapted to the stabilized Nitsche CutFEM of
Burman & Hansbo (2014); see :func:`cutfem.problems.stokes.inf_sup`.
Companion of the center sweep study (velocity block conditioning): same
mesh and cut configurations, pressure side of the stability argument.

Study 1: shift the disk center across one cell width at fixed h.
Study 2: mesh refinement, worst case over shifts per h, with the slope of
         log(min_s beta_h) versus log(h) (about 0: pass, about 1: fail).

Every run is checked: dense size limit, consistency of the coupling blocks
(skew) and constant pressure in the kernel (rq_const).
"""

from dataclasses import replace

import numpy as np

from cutfem import plotting, study
from cutfem.problems import stokes

# Same domain as the center sweep study: 0.5 / h is not an integer, so the
# disk is never tangent to a grid line.
BASE = stokes.Params(gamma_p=0.0, xmin=(-1.243, -1.243), xmax=(1.243, 1.243))
LENGTH = BASE.xmax[0] - BASE.xmin[0]
GAMMA_G_VALUES = [0.0, 0.1, 1.0]

N_CELLS_SWEEP = 32
SHIFTS = np.linspace(0.0, 1.0, 41)          # center shift / h
N_CELLS_REFINE = [16, 24, 32, 48, 64]
N_SHIFTS_REFINE = 20                        # enough to resolve the dips of Study 1
H_REFINE = LENGTH / np.array(N_CELLS_REFINE)

SKEW_TOL = 1e-10        # ||A_pu + A_up^T|| / ||A_up||: pure algebra, round-off only
# Rayleigh quotient of the constant, relative to lam_max. b_h(1, v) = 0 holds
# only up to the consistency of the cut quadratures (divergence theorem between
# dx and ds), so this is not round-off; it shrinks with h. An assembly bug
# (sign, missing boundary term) gives O(1).
RQ_TOL = 1e-4


def inf_sup(n_cells, shift, **overrides):
    """Checked inf-sup data for the disk shifted by ``shift * h`` along x."""
    h = LENGTH / n_cells
    params = replace(BASE, n_cells=n_cells, center=(float(shift * h), 0.0),
                     **overrides)
    with stokes.build(params) as system:
        r = stokes.inf_sup(system)

    where = f"n_cells={n_cells}, s/h={shift:.3f}, {overrides}"
    if not np.isfinite(r["lam_max"]):
        raise RuntimeError(f"dense size limit exceeded ({where})")
    if r["skew"] > SKEW_TOL:
        raise RuntimeError(
            f"inconsistent coupling blocks, skew={r['skew']:.2e} ({where})")
    if abs(r["rq_const"]) > RQ_TOL * r["lam_max"]:
        raise RuntimeError(
            f"constant pressure not in kernel, rq_const={r['rq_const']:.2e} ({where})")
    return r


def _slope(h, values):
    """Slope of log(values) versus log(h); NaN if any value is zero or non-finite."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values) & (values > 0)):
        return np.nan
    return float(np.polyfit(np.log(h), np.log(values), 1)[0])


def _refinement(table, gamma_g):
    """Worst case over shifts for each h, and the slope of min_s beta_h."""
    beta_min, kappa_max, n_small_max = [], [], []
    for n in N_CELLS_REFINE:
        rows = [inf_sup(n, k / N_SHIFTS_REFINE, gamma_g=gamma_g)
                for k in range(N_SHIFTS_REFINE)]
        beta_min.append(min(r["beta"] for r in rows))
        kappa_max.append(max(r["kappa"] for r in rows))
        n_small_max.append(max(r["n_small"] for r in rows))
        table.row([gamma_g, n, beta_min[-1], kappa_max[-1], n_small_max[-1]])
    slope = _slope(H_REFINE, beta_min)
    slope_text = "fail" if np.isnan(slope) else f"slope {slope:.2f}"
    return {"gamma_g": gamma_g, "slope": slope,
            "label": rf"$\gamma_g = {gamma_g:g}$ ({slope_text})",
            "beta_min": beta_min, "kappa_max": kappa_max,
            "n_small_max": n_small_max}


def compute():
    """Shift sweep at fixed h, then worst case over shifts versus h."""
    table = study.Table(["gamma_g", "s/h", "beta", "kappa", "n_small"])
    study.print0("\nStudy 1: center shift sweep")
    table.header()
    sweep = []
    for gamma_g in GAMMA_G_VALUES:
        rows = [inf_sup(N_CELLS_SWEEP, s, gamma_g=gamma_g) for s in SHIFTS]
        for s, r in zip(SHIFTS, rows, strict=True):
            table.row([gamma_g, s, r["beta"], r["kappa"], r["n_small"]])
        sweep.append({"gamma_g": gamma_g, "label": rf"$\gamma_g = {gamma_g:g}$",
                      **{k: np.array([r[k] for r in rows]) for k in rows[0]}})

    table = study.Table(["gamma_g", "n_cells", "beta_min", "kappa_max",
                         "n_small_max"])
    study.print0("\nStudy 2: refinement, worst case over shifts")
    table.header()
    refine = [_refinement(table, gamma_g) for gamma_g in GAMMA_G_VALUES]

    study.print0("Slope of log(min_s beta_h) vs log(h) "
                 "(about 0: pass, about 1: fail, nan: beta_h = 0 on some mesh)")
    for r in refine:
        study.print0(f"  gamma_g = {r['gamma_g']:g}: {r['slope']:.2f}")

    return {"shifts": SHIFTS, "sweep": sweep, "h": H_REFINE, "refine": refine}


def _panels(fig, axes, x, runs, keys, xlabel, log_x):
    colors = plotting.ordered_colors([r["gamma_g"] for r in runs])
    for ax, (key, ylabel) in zip(axes, keys, strict=True):
        for i, (r, color) in enumerate(zip(runs, colors, strict=True)):
            values = np.asarray(r[key], dtype=float)
            ok = np.isfinite(values) & (values > 0)
            ax.plot(x[ok], values[ok], color=color, marker=plotting.MARKERS[i],
                    markevery=None if log_x else 4, label=r["label"])
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