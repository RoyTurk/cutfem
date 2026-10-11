"""Accuracy and conditioning of DFG 2D-1 for a few ghost penalty settings.

Mesh convergence of c_D and c_L, together with the condition number of the
system matrix, for a reduced set of parameters (Nitsche penalty fixed,
gamma_N1 = gamma_N2 = 20):

* gamma_p sweep: gamma_p = 0.001, 0.01, 0.1, 1 with gamma_u = 0.01,
* gamma_u sweep: gamma_u = 0.1, 1 with gamma_p = 0.01,
* baseline: no ghost penalty (gamma_u = gamma_p = 0).

kappa is the 2-norm condition number of the unscaled matrix of the
converged Picard iteration, A(u_h) + G (velocity and pressure, ghost
penalties included), estimated with core.condition_estimate. Each mesh is
built once; the parameters are changed without recompiling the forms.

Usage::

    python parameter_convergence.py
    python parameter_convergence.py --n-y 34 48 --plot-only
"""

import numpy as np
from convergence import ERROR_LABELS, N_Y_FIT, REFERENCE_SLOPES, set_h_ticks

from cutfem import core, plotting, study
from cutfem.problems import dfg2d1

N_Y_SWEEP = [24, 34, 48, 68]
GAMMA_N = 20.0
# (group, gamma_u, gamma_p)
COMBINATIONS = [("gamma_p", 0.01, 0.001), ("gamma_p", 0.01, 0.01),
                ("gamma_p", 0.01, 0.1), ("gamma_p", 0.01, 1.0),
                ("gamma_u", 0.1, 0.01), ("gamma_u", 1.0, 0.01),
                ("baseline", 0.0, 0.0)]
KEYS = [f"err_{q}" for q in dfg2d1.QUANTITIES] + ["kappa"]


def arguments(parser):
    """Mesh levels and Nitsche penalty from the command line."""
    parser.add_argument("--n-y", type=int, nargs="+", default=N_Y_SWEEP,
                        help="cells across the channel (h = 0.41 / n_y)")
    parser.add_argument("--gamma-n", type=float, default=GAMMA_N,
                        help="Nitsche penalty gamma_N1 = gamma_N2")


def measure(system):
    """Solve; return the errors and the condition number of A(u_h) + G."""
    try:
        solution = dfg2d1.solve(system)
    except RuntimeError as error:
        study.print0(f"  {error}")
        return dict.fromkeys(KEYS, np.nan)
    kappa = core.condition_estimate(system.A)[0]
    return {**dfg2d1.errors(system, solution), "kappa": kappa}


def compute(args):
    """Solve every (mesh, combination) and estimate kappa."""
    table = study.Table(["n_y", "h", "gamma_u", "gamma_p", *KEYS], width=12)
    table.header()
    n_ys, h = sorted(args.n_y), []
    runs = [{"group": g, "gamma_u": g_u, "gamma_p": g_p, **{k: [] for k in KEYS}}
            for g, g_u, g_p in COMBINATIONS]

    for n_y in n_ys:
        with dfg2d1.build(dfg2d1.Params(n_y=n_y)) as system:
            h.append(system.h)
            for run in runs:
                dfg2d1.update_parameters(
                    system, reset=True, gamma_N1=args.gamma_n, gamma_N2=args.gamma_n,
                    gamma_u=run["gamma_u"], gamma_p=run["gamma_p"])
                result = measure(system)
                for key in KEYS:
                    run[key].append(result[key])
                table.row([n_y, h[-1], run["gamma_u"], run["gamma_p"],
                           *(result[k] for k in KEYS)])

    fine = np.asarray(n_ys) >= N_Y_FIT
    study.print0(f"\nGeometric mean of the errors over n_y >= {N_Y_FIT}, "
                 f"kappa on the finest mesh:")
    for run in runs:
        means = [np.exp(np.mean(np.log(np.asarray(run[f"err_{q}"])[fine])))
                 for q in dfg2d1.QUANTITIES]
        study.print0(f"  gamma_u = {run['gamma_u']:<5g} gamma_p = {run['gamma_p']:<6g}"
                     f"  c_D {means[0]:.2e}  c_L {means[1]:.2e}"
                     f"  kappa {run['kappa'][-1]:.2e}")
    return {"n_y": n_ys, "h": h, "gamma_N": args.gamma_n, "runs": runs}


def style(run, color):
    """Line style: solid for the gamma_p sweep, dashed for gamma_u, dotted baseline."""
    label = rf"$\gamma_u = {run['gamma_u']:g},\ \gamma_p = {run['gamma_p']:g}$"
    marker, linestyle = {"gamma_p": ("o", "-"), "gamma_u": ("s", "--"),
                         "baseline": ("^", ":")}[run["group"]]
    return {"color": color, "marker": marker, "linestyle": linestyle, "label": label}


def find(data, gamma_u, gamma_p):
    """Return the run with the given ghost penalty parameters."""
    return next(r for r in data["runs"]
                if r["gamma_u"] == gamma_u and r["gamma_p"] == gamma_p)


def plot_errors(data):
    """Relative errors of c_D and c_L versus h, all combinations."""
    fig, axes = plotting.figure(ncols=2, aspect=0.9)
    h = np.asarray(data["h"])
    fine = np.asarray(data["n_y"]) >= N_Y_FIT
    colors = plotting.ordered_colors(data["runs"])
    for ax, q in zip(axes, dfg2d1.QUANTITIES, strict=True):
        for run, color in zip(data["runs"], colors, strict=True):
            ax.loglog(h, run[f"err_{q}"], markersize=3.5, **style(run, color))
        values = np.asarray(find(data, 0.01, 0.01)[f"err_{q}"])
        plotting.slope_triangle(ax, h[fine], values[fine], REFERENCE_SLOPES[q])
        set_h_ticks(ax)
        ax.set_ylabel(ERROR_LABELS[q])
    plotting.legend(axes[0], ncols=3)
    return fig


def plot_conditioning(data):
    """kappa(A) versus h on a broken y-axis: gamma_u = gamma_p = 0 above, gamma_u below.

    gamma_p does not change kappa (the gamma_p sweep curves coincide), so the
    ghost penalty curves vary gamma_u at gamma_p = 0.01.
    """
    fig, (ax_top, ax_bottom) = plotting.figure(
        nrows=2, width=0.62, aspect=0.42, sharex=True,
        gridspec_kw={"height_ratios": [1, 2]})
    fig.get_layout_engine().set(hspace=0.0, h_pad=0.02)
    h = np.asarray(data["h"])
    fine = np.asarray(data["n_y"]) >= N_Y_FIT

    gamma_u = sorted(r["gamma_u"] for r in data["runs"]
                     if r["gamma_p"] == 0.01 and r["gamma_u"] > 0)
    curves = [(find(data, g_u, 0.01)["kappa"], color, plotting.MARKERS[i], "-",
               rf"$\gamma_u = {g_u:g},\ \gamma_p = 0.01$")
              for i, (g_u, color) in enumerate(
                  zip(gamma_u, plotting.ordered_colors(gamma_u), strict=True))]
    baseline = find(data, 0.0, 0.0)["kappa"]
    curves.append((baseline, plotting.CATEGORICAL[len(curves)], "^", ":",
                   r"$\gamma_u = \gamma_p = 0$"))
    for ax in (ax_top, ax_bottom):
        for kappa, color, marker, linestyle, label in curves:
            ax.loglog(h, kappa, color=color, marker=marker, linestyle=linestyle,
                      label=label)

    with_gp = np.concatenate([c[0] for c in curves[:-1]])
    ax_bottom.set_ylim(with_gp.min() / 1.5, with_gp.max() * 1.5)
    ax_top.set_ylim(min(baseline) / 10, max(baseline) * 10)
    plotting.broken_axis(ax_top, ax_bottom)
    kappa = np.asarray(find(data, max(gamma_u), 0.01)["kappa"])
    plotting.slope_triangle(ax_bottom, h[fine], kappa[fine], -2)
    set_h_ticks(ax_bottom)
    fig.supylabel(r"$\kappa(A)$",
                  fontsize=ax_bottom.yaxis.label.get_fontsize())
    plotting.legend(ax_bottom, ncols=2)
    return fig


def plot(data):
    """Errors (parameter_convergence) and kappa (_conditioning)."""
    return {"": plot_errors(data), "conditioning": plot_conditioning(data)}


if __name__ == "__main__":
    study.run("dfg2d1/parameter_convergence", compute, plot, arguments=arguments)
