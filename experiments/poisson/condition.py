"""Condition number of the Poisson CutFEM matrix versus h.

Without ghost penalty the condition number is unbounded in the cut
position; with it, cond(A) = O(h^-2) as for fitted FEM.
"""

from cutfem import core, plotting, study
from cutfem.problems import poisson

N_CELLS = [4, 8, 16, 32, 64]
TAU_VALUES = [0.0, 1.0]
BETA = 30.0


def compute():
    """Refinement sweep with and without ghost penalty."""
    results = []
    for tau in TAU_VALUES:
        study.print0(f"\nbeta = {BETA:g}, tau = {tau:g}")
        columns = study.sweep(
            poisson, poisson.Params(beta=BETA, tau=tau), "n_cells", N_CELLS,
            extras=lambda s: {"cond": core.condition_number(s.A)})
        study.print0(f"  cond orders: {study.rates(columns['h'], columns['cond'])}")
        results.append({"tau": tau, "columns": columns})
    return results


def plot(results):
    """Condition number and L2 error versus h."""
    fig, (ax_cond, ax_l2) = plotting.figure(ncols=2, aspect=0.9)
    colors = plotting.ordered_colors([r["tau"] for r in results])

    for i, (r, color) in enumerate(zip(results, colors, strict=True)):
        label = "no ghost penalty" if r["tau"] == 0 else rf"$\tau = {r['tau']:g}$"
        c = r["columns"]
        ax_cond.loglog(c["h"], c["cond"], color=color,
                       marker=plotting.MARKERS[i], label=label)
        ax_l2.loglog(c["h"], c["L2_u"], color=color, marker=plotting.MARKERS[i])

    ghost = results[-1]["columns"]
    plotting.slope_triangle(ax_cond, ghost["h"], ghost["cond"], -2)
    plotting.slope_triangle(ax_l2, ghost["h"], ghost["L2_u"], 2)

    ax_cond.set_ylabel(r"$\kappa(A)$")
    ax_l2.set_ylabel(r"$\|u - u_h\|_{L^2(\Omega)}$")
    for ax in (ax_cond, ax_l2):
        ax.set_xlabel(r"$h$")
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("poisson/condition", compute, plot)
