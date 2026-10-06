"""Condition number of the Stokes velocity block versus h.

kappa(A_uu) for several ghost penalty parameters. The full matrix A is
singular (constant pressures), so the velocity block is the meaningful
measure of the coercive part.
"""

from cutfem import core, plotting, study
from cutfem.problems import stokes

N_CELLS = [8, 16, 32, 64]
GAMMA_G_VALUES = [0.0, 0.1, 1.0, 10.0]


def conditioning(s):
    """Return the condition number of the velocity block."""
    return {"cond": core.condition_number(s.A_uu)}


def compute():
    """Refinement sweep for each gamma_g."""
    results = []
    for gamma_g in GAMMA_G_VALUES:
        study.print0(f"\ngamma_g = {gamma_g:g}")
        columns = study.sweep(stokes, stokes.Params(gamma_g=gamma_g), "n_cells",
                              N_CELLS, extras=conditioning)
        results.append({"gamma_g": gamma_g, "columns": columns})
    return results


def plot(results):
    """kappa(A_uu) versus h, one curve per gamma_g."""
    fig, ax = plotting.figure(width=0.7, aspect=0.75)
    colors = plotting.ordered_colors([r["gamma_g"] for r in results])
    for i, (r, color) in enumerate(zip(results, colors, strict=True)):
        ax.loglog(r["columns"]["h"], r["columns"]["cond"], color=color,
                  marker=plotting.MARKERS[i], label=rf"$\gamma_g = {r['gamma_g']:g}$")
    for r, color in zip(results, colors, strict=True):
        plotting.mark_nonfinite(ax, r["columns"]["h"], r["columns"]["cond"], color)

    best = min(results, key=lambda r: r["columns"]["cond"][-1])["columns"]
    plotting.slope_triangle(ax, best["h"], best["cond"], -2)
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\kappa(A_{uu})$")
    plotting.legend_above(fig, ncols=2)
    return fig


if __name__ == "__main__":
    study.run("stokes/condition", compute, plot)
