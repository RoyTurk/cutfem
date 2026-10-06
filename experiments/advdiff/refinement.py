"""Mesh refinement for the advection-diffusion CutFEM solver.

Profile of u_h along y = 0.5 at fixed, small diffusion (advection dominated,
Pe_L = |beta| L / mu = 200) for several meshes, with and without CIP. The
coarsest and finest solutions are written for ParaView.
"""

from dataclasses import replace

import numpy as np

from cutfem import plotting, study
from cutfem.problems import advdiff

BASE = advdiff.Params(mu=5e-3)
N_CELLS = [64, 128, 256, 512]
GAMMA_BETA_VALUES = [0.0, 0.5]
Y_LINE = 0.5
X_LINE = np.linspace(BASE.xmin[0], BASE.xmax[0], 1001)


def compute():
    """Line profiles for every (gamma_beta, n_cells)."""
    table = study.Table(["gamma_beta", "n_cells", "Pe_h", "min u_h", "max u_h"])
    table.header()
    runs = []
    for gamma_beta in GAMMA_BETA_VALUES:
        profiles, peclet = [], []
        for n in N_CELLS:
            params = replace(BASE, n_cells=n, gamma_beta=gamma_beta)
            with advdiff.build(params) as system:
                uh = advdiff.solve(system)
                line = advdiff.sample_line(system, uh, X_LINE, Y_LINE)
                peclet.append(advdiff.peclet_h(params, system.h))
                if n in (N_CELLS[0], N_CELLS[-1]):
                    path = (study.RESULTS_DIR / "advdiff"
                            / f"refinement_n{n}_gb{gamma_beta:g}.pvd")
                    advdiff.save_vtk(path, system, uh)
            profiles.append(line)
            table.row([gamma_beta, n, peclet[-1], np.nanmin(line), np.nanmax(line)])
        runs.append({"gamma_beta": gamma_beta, "profiles": profiles,
                     "peclet_h": peclet})
    return {"x": X_LINE, "n_cells": N_CELLS, "runs": runs}


def plot(data):
    """One panel per stabilization, one curve per mesh."""
    fig, axes = plotting.figure(nrows=2, aspect=0.4, sharex=True, sharey=True)
    colors = plotting.ordered_colors(data["n_cells"])
    for ax, run in zip(axes, data["runs"], strict=True):
        for color, n, peclet, profile in zip(colors, data["n_cells"],
                                             run["peclet_h"], run["profiles"],
                                             strict=True):
            ax.plot(data["x"], profile, color=color,
                    label=rf"$n = {n}$ ($Pe_h = {peclet:.3g}$)")
        ax.axvspan(*advdiff.disk_chord(BASE, Y_LINE), color="0.9", zorder=0)
        name = "no CIP" if run["gamma_beta"] == 0 else "CIP"
        plotting.panel_label(ax, rf"{name}, $\gamma_\beta = {run['gamma_beta']:g}$")
        ax.set_ylabel(rf"$u_h(x, {Y_LINE:g})$")
    axes[-1].set_xlabel(r"$x$")
    plotting.legend_above(fig, ncols=2)
    return fig


if __name__ == "__main__":
    study.run("advdiff/refinement", compute, plot)
