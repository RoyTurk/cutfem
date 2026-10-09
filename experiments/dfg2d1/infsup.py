"""Numerical inf-sup test for the DFG 2D-1 CutFEM discretization.

Same test as experiments/stokes/infsup.py (Chapelle & Bathe, adapted to the
stabilized Nitsche CutFEM), on the channel-cylinder geometry:

    beta_h^2 = lambda_min of (B N^-1 B^T + G_p) q = lambda M q,

see dfg2d1.inf_sup_matrices and core.inf_sup_eigenvalues (sparse version,
no dense Schur complement). The do-nothing outflow fixes the pressure level,
so the constant pressure is not in the kernel. The test only involves the
pressure-velocity coupling, not the flow solution.

Study 1: cut sweep. n_y = 24, ..., 40: h barely changes, the cut
         configuration changes completely from one mesh to the next.
Study 2: refinement. For n_y in N_Y_LEVELS, the worst case over the five
         neighbouring meshes n_y - 2, ..., n_y + 2, and the slope of
         log(min beta_h) versus log(h) (about 0: pass).

Usage::

    python infsup.py
    python infsup.py --plot-only
"""

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import dfg2d1

GAMMA_P = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
N_Y_SWEEP = list(range(24, 41))
N_Y_LEVELS = [17, 24, 34, 48, 68]
WINDOW = 2


def window(n_y):
    """Return the neighbouring meshes of a refinement level."""
    return range(n_y - WINDOW, n_y + WINDOW + 1)


def compute():
    """Eigenvalues for every mesh and gamma_p, then sweep and refinement data."""
    meshes = sorted(set(N_Y_SWEEP).union(*(window(n) for n in N_Y_LEVELS)))
    table = study.Table(["n_y", "h", "gamma_p", "beta", "lam_max", "kappa"])
    table.header()
    results = {}
    for n_y in meshes:
        with dfg2d1.build(dfg2d1.Params(n_y=n_y)) as system:
            N, A_up, G_1, M = dfg2d1.inf_sup_matrices(system)
            rows = []
            for gamma_p in GAMMA_P:
                G = G_1.copy()
                G.scale(gamma_p)
                lam_min, lam_max = core.inf_sup_eigenvalues(N, A_up, G, M)
                G.destroy()
                beta = np.sqrt(lam_min)
                kappa = lam_max / lam_min if lam_min > 0 else np.inf
                rows.append({"beta": beta, "kappa": kappa})
                table.row([n_y, system.h, gamma_p, beta, lam_max, kappa])
            results[n_y] = {"h": system.h, "rows": rows}
            for mat in (N, A_up, G_1, M):
                mat.destroy()

    def column(n_ys, i, key):
        return [results[n]["rows"][i][key] for n in n_ys]

    sweep = [{"gamma_p": g, "beta": column(N_Y_SWEEP, i, "beta"),
              "kappa": column(N_Y_SWEEP, i, "kappa")}
             for i, g in enumerate(GAMMA_P)]
    refine = []
    for i, g in enumerate(GAMMA_P):
        beta_min = [min(column(window(n), i, "beta")) for n in N_Y_LEVELS]
        kappa_max = [max(column(window(n), i, "kappa")) for n in N_Y_LEVELS]
        refine.append({"gamma_p": g, "beta_min": beta_min, "kappa_max": kappa_max})
    h_levels = [results[n]["h"] for n in N_Y_LEVELS]

    study.print0("\nSlope of log(min beta_h) vs log(h) (about 0: pass):")
    for r in refine:
        r["slope"] = _slope(h_levels, r["beta_min"])
        study.print0(f"  gamma_p = {r['gamma_p']:g}: {r['slope']:.2f}")
    return {"n_y": N_Y_SWEEP, "sweep": sweep, "h": h_levels, "refine": refine}


def _slope(h, values):
    """Slope of log(values) versus log(h); NaN if a value is zero."""
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values) & (values > 0)):
        return np.nan
    return float(np.polyfit(np.log(h), np.log(values), 1)[0])


def _panels(fig, axes, x, runs, keys, xlabel, log_x, slope=False):
    colors = plotting.ordered_colors([r["gamma_p"] for r in runs])
    for ax, (key, ylabel) in zip(axes, keys, strict=True):
        for i, (r, color) in enumerate(zip(runs, colors, strict=True)):
            values = np.asarray(r[key], dtype=float)
            ok = np.isfinite(values) & (values > 0)
            label = rf"$\gamma_p = {r['gamma_p']:g}$"
            if slope:
                text = "fail" if np.isnan(r["slope"]) else f"slope {r['slope']:.2f}"
                label += f" ({text})"
            ax.plot(np.asarray(x)[ok], values[ok], color=color,
                    marker=plotting.MARKERS[i], markersize=3.5, label=label)
        ax.set_yscale("log")
        ax.set_xscale("log" if log_x else "linear")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for r, color in zip(runs, colors, strict=True):   # after all curves
            plotting.mark_nonfinite(ax, x, r[key], color)
    plotting.legend_above(fig, ax=axes[0], ncols=3)


def plot(data):
    """Cut sweep figure and refinement figure (beta_h and kappa)."""
    fig_sweep, axes = plotting.figure(ncols=2, aspect=0.85)
    _panels(fig_sweep, axes, data["n_y"], data["sweep"],
            [("beta", r"$\beta_h$"), ("kappa", r"$\lambda_{\max} / \lambda_{\min}$")],
            r"cells across the channel $n_y$", log_x=False)

    fig_refine, axes = plotting.figure(ncols=2, aspect=0.85)
    _panels(fig_refine, axes, data["h"], data["refine"],
            [("beta_min", r"$\min \beta_h$ over 5 meshes"),
             ("kappa_max", r"$\max \lambda_{\max} / \lambda_{\min}$")],
            r"$h$", log_x=True, slope=True)
    return {"sweep": fig_sweep, "refinement": fig_refine}


if __name__ == "__main__":
    study.run("dfg2d1/infsup", compute, plot)
