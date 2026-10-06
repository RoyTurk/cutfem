"""Peclet sweep for the advection-diffusion CutFEM solver.

Profile of u_h along y = 0.5, from diffusion to advection dominated flow
(grid Peclet Pe_h = |beta| h / (2 mu) from 0.1 to 100 on a fixed mesh),
with and without CIP stabilization.
"""

from dataclasses import replace

import numpy as np

from cutfem import plotting, study
from cutfem.problems import advdiff

BASE = advdiff.Params(n_cells=64)
PECLET_H = [0.1, 1.0, 10.0, 100.0]
GAMMA_BETA_VALUES = [0.0, 0.5]
Y_LINE = 0.5
X_LINE = np.linspace(BASE.xmin[0], BASE.xmax[0], 1001)


def mu_for(peclet_h):
    """Diffusion giving the grid Peclet number ``peclet_h`` on BASE's mesh."""
    h = (BASE.xmax[0] - BASE.xmin[0]) / BASE.n_cells
    return float(np.linalg.norm(BASE.beta)) * h / (2.0 * peclet_h)


def compute():
    """Line profiles for every (gamma_beta, Pe_h)."""
    table = study.Table(["gamma_beta", "Pe_h", "mu", "min u_h", "max u_h"])
    table.header()
    runs = []
    for gamma_beta in GAMMA_BETA_VALUES:
        profiles = []
        for peclet in PECLET_H:
            params = replace(BASE, mu=mu_for(peclet), gamma_beta=gamma_beta)
            with advdiff.build(params) as system:
                line = advdiff.sample_line(system, advdiff.solve(system),
                                           X_LINE, Y_LINE)
            profiles.append(line)
            table.row([gamma_beta, peclet, params.mu, np.nanmin(line),
                       np.nanmax(line)])
        runs.append({"gamma_beta": gamma_beta, "profiles": profiles})
    return {"x": X_LINE, "peclet_h": PECLET_H, "runs": runs}


def plot(data):
    """One panel per stabilization, one curve per Peclet number."""
    fig, axes = plotting.figure(nrows=2, aspect=0.4, sharex=True, sharey=True)
    colors = plotting.ordered_colors(data["peclet_h"])
    for ax, run in zip(axes, data["runs"], strict=True):
        for color, peclet, profile in zip(colors, data["peclet_h"], run["profiles"],
                                          strict=True):
            ax.plot(data["x"], profile, color=color, label=rf"$Pe_h = {peclet:g}$")
        ax.axvspan(*advdiff.disk_chord(BASE, Y_LINE), color="0.9", zorder=0)
        name = "no CIP" if run["gamma_beta"] == 0 else "CIP"
        plotting.panel_label(ax, rf"{name}, $\gamma_\beta = {run['gamma_beta']:g}$")
        ax.set_ylabel(rf"$u_h(x, {Y_LINE:g})$")
    axes[-1].set_xlabel(r"$x$")
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run("advdiff/peclet_sweep", compute, plot)
