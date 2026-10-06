"""Sensitivity of the Stokes CutFEM errors to the stabilization parameters.

Each of gamma_mu, gamma_g and gamma_p is swept over several decades while
the other two stay at their baseline values.
"""

import numpy as np

from cutfem import plotting, study
from cutfem.problems import stokes

BASE = stokes.Params(n_cells=32, gamma_mu=10.0, gamma_g=0.1, gamma_p=0.0)
SWEEPS = {
    "gamma_mu": (np.logspace(-2, 3, 20), r"\gamma_\mu"),
    "gamma_g": (np.logspace(-4, 3, 20), r"\gamma_g"),
    "gamma_p": (np.logspace(-4, 3, 20), r"\gamma_p"),
}


def compute():
    """One sweep per stabilization parameter."""
    results = {}
    for name, (values, _) in SWEEPS.items():
        study.print0(f"\nSweeping {name}")
        results[name] = study.sweep(stokes, BASE, name, values)
    return results


def plot(results):
    """Velocity (solid) and pressure (dashed) L2 errors versus each parameter."""
    fig, ax = plotting.figure(width=0.8, aspect=0.7)
    for i, (name, (_, symbol)) in enumerate(SWEEPS.items()):
        c = results[name]
        color = plotting.CATEGORICAL[i]
        ax.loglog(c[name], c["L2_u"], color=color, marker=plotting.MARKERS[i],
                  markevery=3, label=rf"${symbol}$, velocity")
        ax.loglog(c[name], c["L2_p"], color=color, marker=plotting.MARKERS[i],
                  markevery=3, linestyle="--", label=rf"${symbol}$, pressure")
    ax.set_xlabel(r"stabilization parameter $\gamma$")
    ax.set_ylabel(r"$L^2(\Omega)$ error")
    plotting.legend_above(fig, ncols=3)
    return fig


if __name__ == "__main__":
    study.run("stokes/parameter", compute, plot)
