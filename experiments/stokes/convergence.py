"""Mesh convergence of the Stokes CutFEM solver (Q2/Q1).

Velocity L2 and H1 errors, pressure L2 error and divergence residual
versus h, with and without ghost penalty and Nitsche penalty.
"""

from cutfem import plotting, study
from cutfem.problems import stokes

N_CELLS = [8, 16, 32, 64, 128, 256]
CONFIGS = [
    dict(gamma_mu=10.0, gamma_g=0.0),
    dict(gamma_mu=10.0, gamma_g=0.1),
    dict(gamma_mu=0.0, gamma_g=0.1),
]
PANELS = [
    ("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$"),
    ("H1_u", r"$|u - u_h|_{H^1(\Omega)}$"),
    ("L2_p", r"$\|p - p_h\|_{L^2(\Omega)}$"),
    ("div_u", r"$\|\nabla \cdot u_h\|_{L^2(\Omega)}$"),
]


def compute():
    """Refinement sweep for every configuration."""
    results = []
    for config in CONFIGS:
        params = stokes.Params(**config)
        study.print0(f"\ngamma_mu = {params.gamma_mu:g}, gamma_g = {params.gamma_g:g}")
        columns = study.sweep(stokes, params, "n_cells", N_CELLS)
        study.print_rates(columns, stokes.expected_rates(params))
        results.append({**config, "columns": columns})
    return results


def plot(results):
    """One panel per error quantity, one curve per configuration."""
    fig, axes = plotting.figure(nrows=2, ncols=2, aspect=0.8)
    rates = stokes.expected_rates(stokes.Params())

    for ax, (key, ylabel) in zip(axes.flat, PANELS, strict=True):
        for i, r in enumerate(results):
            label = rf"$\gamma_\mu = {r['gamma_mu']:g},\ \gamma_g = {r['gamma_g']:g}$"
            ax.loglog(r["columns"]["h"], r["columns"][key],
                      marker=plotting.MARKERS[i], label=label)
        lowest = min(results, key=lambda r: r["columns"][key][-1])["columns"]
        plotting.slope_triangle(ax, lowest["h"], lowest[key], rates[key])
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)

    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("stokes/convergence", compute, plot)
