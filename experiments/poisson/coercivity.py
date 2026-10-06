"""Coercivity threshold of the Nitsche penalty beta for the Poisson CutFEM.

Counts negative eigenvalues of A for a range of beta, with and without ghost
penalty: the ghost penalty makes the coercivity threshold independent of the
cut.
"""

import numpy as np

from cutfem import core, plotting, study
from cutfem.problems import poisson

BETA_VALUES = [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
TAU_VALUES = [0.0, 0.1, 1.0]
N_CELLS = 32


def threshold(columns):
    """Smallest beta from which A is coercive (no negative eigenvalue)."""
    coercive = columns["n_neg"] == 0
    return float(columns["beta"][np.argmax(coercive)]) if coercive.any() else None


def spectrum(s):
    """Return the number of negative eigenvalues and the smallest one of A."""
    eigs = core.eigenvalues(s.A)
    return {"n_neg": int(np.sum(eigs < 0)), "min_eig": eigs[0]}


def compute():
    """Beta sweep for each tau."""
    results = []
    for tau in TAU_VALUES:
        study.print0(f"\ntau = {tau:g}")
        columns = study.sweep(poisson, poisson.Params(n_cells=N_CELLS, tau=tau),
                              "beta", BETA_VALUES, extras=spectrum)
        study.print0(f"  coercivity threshold: beta >= {threshold(columns)}")
        results.append({"tau": tau, "columns": columns})
    return results


def plot(results):
    """Negative eigenvalues, smallest eigenvalue and L2 error versus beta."""
    fig, axes = plotting.figure(ncols=3, aspect=1.0)
    colors = plotting.ordered_colors([r["tau"] for r in results])
    panels = [("n_neg", "negative eigenvalues", "linear"),
              ("min_eig", r"$\lambda_{\min}(A)$", "linear"),
              ("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$", "log")]

    for ax, (key, ylabel, scale) in zip(axes, panels, strict=True):
        for i, (r, color) in enumerate(zip(results, colors, strict=True)):
            ax.plot(r["columns"]["beta"], r["columns"][key], color=color,
                    marker=plotting.MARKERS[i], label=rf"$\tau = {r['tau']:g}$")
        ax.set_xscale("log")
        ax.set_yscale(scale)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
    for ax in axes[:2]:
        ax.axhline(0.0, color=plotting.INK, linewidth=0.6)

    beta_0 = threshold(results[0]["columns"])
    if beta_0 is not None:
        axes[0].axvline(beta_0, color=colors[0], linestyle=":", linewidth=1.0)
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("poisson/coercivity", compute, plot)
