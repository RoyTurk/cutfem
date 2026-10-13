"""Summary of the DFG 2D-1 study: the role of the ghost and Nitsche penalties.

Four panels from the saved studies (``parameter_convergence``, ``condition``)
plus the one missing case, the accuracy without ghost penalty, computed here:

(a) drag error versus h for gamma_u = gamma_p = 0, 0.01, 0.1, 1;
(b) lift error versus h, same cases;
(c) kappa(A) versus h, same cases (broken y-axis);
(d) drag error versus h for gamma_N = 10, 30, 100 at gamma_u = gamma_p = 0.01.

gamma_N = 30 unless varied. Run ``parameter_convergence.py`` and
``condition.py`` first.
"""

import dataclasses

import numpy as np

from cutfem import navier_stokes as ns
from cutfem import plotting, study
from cutfem.problems import dfg2d1

GAMMA_N = 30.0
GAMMA_G = [0.0, 0.01, 0.1, 1.0]     # none, weak (best), default, strong
GAMMA_G_NITSCHE = 0.01              # ghost penalty of panel (d)
REFERENCE_ORDER = 4


def unstabilized_accuracy(n_y):
    """c_D and c_L errors without ghost penalty (the case missing elsewhere).

    When Newton stalls (residual floor of the ill-conditioned Jacobian), the
    last iterate is kept: its errors and kappa are recorded, flagged by
    ``stalled``, since the bad cut is precisely what the figure shows.
    """
    params = dataclasses.replace(dfg2d1.Params(), gamma_N=GAMMA_N,
                                 gamma_u=0.0, gamma_p=0.0)
    rows, table = [], None
    for n in n_y:
        with dfg2d1.build(dataclasses.replace(params, n_y=n)) as system:
            row = {"n_y": n, "h": system.h, "stalled": 0, "kappa": np.nan}
            try:
                solution = dfg2d1.solve(system)
            except RuntimeError as error:
                study.print0(f"  n_y = {n}: {error}; keeping the last iterate")
                solution = (system.uh, system.ph)
                ns.assemble_jacobian(system)            # Jacobian at that iterate
                row["stalled"] = 1
                row["kappa"] = ns.condition_number(system)[0]
            row.update(dfg2d1.errors(system, solution))
        if table is None:
            table = study.Table(row)
            table.header()
        table.row(row)
        rows.append(row)
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def compute():
    """Collect the curves of the four panels; compute gamma_u = gamma_p = 0."""
    sweep = study.load_data("dfg2d1/parameter_convergence")
    conditioning = study.load_data("dfg2d1/condition")

    def errors(gamma_N, gamma_g):
        for run in sweep["runs"]:
            if run["gamma_N"] == gamma_N and run["gamma_g"] == gamma_g:
                return run["columns"]
        raise KeyError((gamma_N, gamma_g))

    n_y = list(errors(GAMMA_N, GAMMA_G[1])["n_y"])
    study.print0(f"gamma_N = {GAMMA_N:g}, gamma_u = gamma_p = 0 (accuracy)")
    accuracy = {0.0: unstabilized_accuracy(n_y)}
    accuracy.update({g: errors(GAMMA_N, g) for g in GAMMA_G if g > 0.0})
    kappa = {run["gamma_g"]: dict(run["columns"]) for run in conditioning["runs"]
             if run["gamma_g"] in GAMMA_G}
    # The condition study has no kappa where Newton stalled: use the stalled iterate's
    stalled = np.asarray(accuracy[0.0]["stalled"], dtype=bool)
    kappa_0 = np.array(kappa[0.0]["kappa"], dtype=float)
    kappa_0[stalled] = accuracy[0.0]["kappa"][stalled]
    kappa[0.0]["kappa"] = kappa_0
    kappa[0.0]["stalled"] = accuracy[0.0]["stalled"]
    nitsche = {gN: errors(gN, GAMMA_G_NITSCHE) for gN in sorted(
        {run["gamma_N"] for run in sweep["runs"]})}
    return {"accuracy": [{"gamma_g": g, "columns": accuracy[g]} for g in GAMMA_G],
            "kappa": [{"gamma_g": g, "columns": kappa[g]} for g in GAMMA_G],
            "nitsche": [{"gamma_N": gN, "columns": c} for gN, c in nitsche.items()]}


def reference_line(ax, curves, order):
    """Dashed C h^order over the whole h range, just below all curves."""
    h = np.asarray(curves[0][0])
    ratios = [np.asarray(y) / h ** order for _, y in curves]
    C = 0.5 * np.nanmin(ratios)
    ax.loglog(h, C * h ** order, color=plotting.INK, linestyle="--", linewidth=0.8)


def mark_stalled(ax, columns, key, color, x="h"):
    """Open marker on the points from a stalled Newton solve (last iterate)."""
    stalled = np.asarray(columns.get("stalled", []), dtype=bool)
    if stalled.any():
        ax.scatter(np.asarray(columns[x])[stalled], np.asarray(columns[key])[stalled],
                   s=90, facecolors="white", edgecolors=color, linewidths=1.2, zorder=5)


def broken_axis(ax_top, ax_bottom):
    """Join two stacked axes into one panel with a broken y-axis (// marks)."""
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    marks = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle="none",
                 color=plotting.INK, mec=plotting.INK, mew=0.6, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **marks)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **marks)


def plot(data):
    """2 x 2 panels: drag, lift, conditioning, Nitsche penalty."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator

    fig = plt.figure(figsize=(plotting.TEXTWIDTH, 0.82 * plotting.TEXTWIDTH))
    grid = fig.add_gridspec(2, 2)
    ax_drag = fig.add_subplot(grid[0, 0])
    ax_lift = fig.add_subplot(grid[0, 1], sharex=ax_drag)
    cell = grid[1, 0].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.0)
    ax_k_top = fig.add_subplot(cell[0], sharex=ax_drag)
    ax_k = fig.add_subplot(cell[1], sharex=ax_drag)
    ax_nitsche = fig.add_subplot(grid[1, 1], sharex=ax_drag)
    style = {g: (plotting.CATEGORICAL[i], plotting.MARKERS[i])
             for i, g in enumerate(GAMMA_G)}

    # (a), (b): accuracy versus the ghost penalty
    for ax, key in ((ax_drag, "err_c_D"), (ax_lift, "err_c_L")):
        curves = []
        for run in data["accuracy"]:
            color, marker = style[run["gamma_g"]]
            c = run["columns"]
            ax.loglog(c["h"], c[key], color=color, marker=marker)
            curves.append((c["h"], c[key]))
        reference_line(ax, curves, REFERENCE_ORDER)
        for run in data["accuracy"]:
            mark_stalled(ax, run["columns"], key, style[run["gamma_g"]][0])
    ax_drag.set_ylabel(r"$|c_D - c_D^{\mathrm{ref}}| / |c_D^{\mathrm{ref}}|$")
    ax_lift.set_ylabel(r"$|c_L - c_L^{\mathrm{ref}}| / |c_L^{\mathrm{ref}}|$")
    plotting.panel_label(ax_drag, "(a) drag")
    plotting.panel_label(ax_lift, "(b) lift")

    # (c): conditioning, unstabilized curve above the break
    curves = []
    for run in data["kappa"]:
        color, marker = style[run["gamma_g"]]
        c = run["columns"]
        ax = ax_k_top if run["gamma_g"] == 0.0 else ax_k
        ax.loglog(c["h"], c["kappa"], color=color, marker=marker)
        if run["gamma_g"] > 0.0:
            curves.append((c["h"], c["kappa"]))
    reference_line(ax_k, curves, -2)
    broken_axis(ax_k_top, ax_k)
    ax_k_top.yaxis.set_major_locator(LogLocator(base=1e4))
    mark_stalled(ax_k_top, data["kappa"][0]["columns"], "kappa", style[0.0][0])
    ax_k.set_ylabel(r"$\kappa(A)$")
    ax_k.yaxis.set_label_coords(-0.16, 0.75)
    plotting.panel_label(ax_k_top, "(c) condition number")

    # (d): Nitsche penalty at gamma_u = gamma_p = 0.01
    curves = []
    for i, run in enumerate(data["nitsche"]):
        c = run["columns"]
        ax_nitsche.loglog(c["h"], c["err_c_D"], color=plotting.CATEGORICAL[4 + i],
                          marker=plotting.MARKERS[(4 + i) % len(plotting.MARKERS)],
                          label=rf"$\gamma_N = {run['gamma_N']:g}$")
        curves.append((c["h"], c["err_c_D"]))
    reference_line(ax_nitsche, curves, REFERENCE_ORDER)
    ax_nitsche.set_ylabel(r"$|c_D - c_D^{\mathrm{ref}}| / |c_D^{\mathrm{ref}}|$")
    plotting.panel_label(ax_nitsche,
                         rf"(d) drag, $\gamma_u = \gamma_p = {GAMMA_G_NITSCHE:g}$")
    ax_nitsche.legend(loc="lower right")

    for ax in (ax_k, ax_nitsche):
        ax.set_xlabel(r"$h$")
    for ax in (ax_drag, ax_lift, ax_k_top):
        ax.tick_params(axis="x", which="both", labelbottom=False)

    handles = [Line2D([], [], color=style[g][0], marker=style[g][1],
                      label=rf"$\gamma_u = \gamma_p = {g:g}$") for g in GAMMA_G]
    handles += [Line2D([], [], color=plotting.INK, linestyle="--", linewidth=0.8,
                       label=r"$\mathcal{O}(h^{4})$, $\mathcal{O}(h^{-2})$"),
                Line2D([], [], color=style[0.0][0], marker="o", markersize=8,
                       markerfacecolor="white", linestyle="none",
                       label="Newton stalled (last iterate)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    return fig


if __name__ == "__main__":
    study.run("dfg2d1/summary", compute, plot)
