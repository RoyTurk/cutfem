"""Mesh convergence of the Poisson CutFEM solver.

L2 and H1-seminorm errors versus h for Nitsche only, Nitsche with ghost
penalty, and ghost penalty only.
"""

from cutfem import plotting, study
from cutfem.problems import poisson

N_CELLS = [8, 16, 32, 64, 128, 256]
CONFIGS = [
    dict(beta=10.0, tau=0.0),
    dict(beta=10.0, tau=1.0),
    dict(beta=0.0, tau=1.0),
]


def compute():
    """Refinement sweep for every configuration."""
    results = []
    for config in CONFIGS:
        params = poisson.Params(**config)
        study.print0(f"\nbeta = {params.beta:g}, tau = {params.tau:g}")
        columns = study.sweep(poisson, params, "n_cells", N_CELLS)
        study.print_rates(columns, poisson.expected_rates(params))
        results.append({**config, "columns": columns})
    return results


def plot(results):
    """L2 and H1 errors versus h, one curve per configuration."""
    fig, (ax_l2, ax_h1) = plotting.figure(ncols=2, aspect=0.9)
    panels = [(ax_l2, "L2_u", 2, r"$\|u - u_h\|_{L^2(\Omega)}$"),
              (ax_h1, "H1_u", 1, r"$|u - u_h|_{H^1(\Omega)}$")]

    for i, r in enumerate(results):
        label = rf"$\beta = {r['beta']:g},\ \tau = {r['tau']:g}$"
        for ax, key, _, _ in panels:
            ax.loglog(r["columns"]["h"], r["columns"][key],
                      marker=plotting.MARKERS[i], label=label)

    for ax, key, order, ylabel in panels:
        lowest = min(results, key=lambda r: r["columns"][key][-1])["columns"]
        plotting.slope_triangle(ax, lowest["h"], lowest[key], order)
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)

    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("poisson/convergence", compute, plot)
