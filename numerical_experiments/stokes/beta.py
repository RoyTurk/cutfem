"""
stokes_h_sweep.py
Mesh-refinement study for the Stokes CutFEM solver: both Brezzi conditions
versus h, with the cut position varied inside one cell at every h.

  coercivity side : kappa(A_uu)                        expected ~ h^-2
  inf-sup side    : beta_h                             expected ~ h^0
                    kappa_p = cond(B S^-1 B^T + C, T)  expected ~ h^0

Stability must hold uniformly in h AND in the cut position, so for every h the
worst case over the sampled cut positions is plotted (max kappa, min beta,
max kappa_p), with the min-max band over cut positions shaded.
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stokes_problem import build_system, compute_condition
from infsup_test import compute_inf_sup

# Parameters (same physical setup as stokes_center_sweep.py)
radius = 0.5
mu = 1.0
gamma_mu = 10
gamma_p = 0.0
v_degree = 2
p_degree = 1

xmin = np.array([-1.243, -1.243])
xmax = np.array([1.243, 1.243])

gamma_g_values = [0.0, 0.1, 1.0]
n_cells_values = [8, 16, 32, 64]
n_shifts = 8                        # cut positions per mesh
base_center = np.array([0.0, 0.0])

GOLDEN = 0.6180339887498949


def cut_offsets(h, n_shifts):
    """
    Sub-cell shifts of the disk center. A shift by a whole cell reproduces the
    same cut pattern, so all distinct configurations live in [0, h)^2:
    x is spread uniformly, y follows a golden-ratio sequence, which avoids
    symmetric (duplicate) configurations and varies the cut in both directions.
    """
    s = np.arange(n_shifts)
    return np.column_stack([s / n_shifts * h, ((s * GOLDEN + 0.5) % 1.0) * h])


def log_slopes(hs, y):
    """Observed rates between consecutive meshes, d log y / d log h."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(y)) / np.diff(np.log(hs))


def run_study():
    keys = ["cond", "beta", "kappa_p", "n_zero"]
    shape = (len(n_cells_values), n_shifts)
    results = {gg: {k: np.full(shape, np.nan) for k in keys}
               for gg in gamma_g_values}
    hs = (xmax[0] - xmin[0]) / np.array(n_cells_values, dtype=float)

    print("=" * 75)
    print("Stokes CutFEM — mesh refinement study (coercivity + inf-sup)")
    print(f"  radius={radius}, mu={mu}, gamma_mu={gamma_mu}, gamma_p={gamma_p}")
    print(f"  Q{v_degree}/Q{p_degree}, gamma_g values: {gamma_g_values}")
    print(f"  n_cells: {n_cells_values}, {n_shifts} cut positions per mesh")
    print("=" * 75)

    for gg in gamma_g_values:
        print(f"\ngamma_g = {gg}")
        print(f"  {'n':>4} {'h':>8} {'max cond':>11} {'min beta':>11}"
              f" {'max kappa_p':>12} {'max nz':>7}")
        print("  " + "-" * 58)
        for i, n in enumerate(n_cells_values):
            for s, offset in enumerate(cut_offsets(hs[i], n_shifts)):
                sys = build_system(
                    n_cells  = n,
                    center   = base_center + offset,
                    gamma_mu = gamma_mu,
                    gamma_g  = gg,
                    gamma_p  = gamma_p,
                    radius   = radius,
                    xmin     = xmin,
                    xmax     = xmax,
                    mu       = mu,
                    v_degree = v_degree,
                    p_degree = p_degree,
                )
                # both on the unpinned system (no solve in this study)
                results[gg]["cond"][i, s] = compute_condition(sys)
                isup = compute_inf_sup(sys, mu=mu)
                results[gg]["beta"][i, s] = isup["beta"]
                results[gg]["kappa_p"][i, s] = isup["kappa_p"]
                results[gg]["n_zero"][i, s] = isup["n_zero"]

            r = results[gg]
            print(f"  {n:>4d} {hs[i]:>8.4f} {np.max(r['cond'][i]):>11.3e}"
                  f" {np.min(r['beta'][i]):>11.3e} {np.max(r['kappa_p'][i]):>12.3e}"
                  f" {int(np.nanmax(r['n_zero'][i])):>7d}")

    # Observed rates of the worst cases between consecutive meshes
    print("\nObserved rates  d log(worst case) / d log h   (between consecutive meshes)")
    for gg in gamma_g_values:
        r = results[gg]
        print(f"  gamma_g={gg}:"
              f"  cond {np.round(log_slopes(hs, r['cond'].max(axis=1)), 2)}"
              f"  beta {np.round(log_slopes(hs, r['beta'].min(axis=1)), 2)}"
              f"  kappa_p {np.round(log_slopes(hs, r['kappa_p'].max(axis=1)), 2)}")
    return hs, results


# Plotting
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
results_dir = Path("results")


def plot_vs_h(hs, results, key, worst, ylabel, filename,
              ref_slope=None, ref_label=None):
    """
    One curve per gamma_g: worst case over cut positions vs h (log-log),
    shaded band = min..max over cut positions.
    worst: "max" (condition numbers) or "min" (inf-sup constant).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    anchor = None
    for i, gg in enumerate(gamma_g_values):
        vals = results[gg][key]
        vals = np.where(np.isfinite(vals) & (vals > 0), vals, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo, hi = np.nanmin(vals, axis=1), np.nanmax(vals, axis=1)
        wc = hi if worst == "max" else lo
        ax.fill_between(hs, lo, hi, color=f"C{i}", alpha=0.15, linewidth=0)
        ax.loglog(hs, wc, f"{markers[i % len(markers)]}-", color=f"C{i}",
                  linewidth=1.6, markersize=6,
                  label=rf"$\gamma_g={gg}$ ({worst} over cuts)")
        if gg != 0.0 and np.isfinite(wc[0]):
            anchor = wc[0]                    # reference anchored on a stabilized curve

    if ref_slope is not None and anchor is not None:
        ax.loglog(hs, anchor * (hs / hs[0]) ** ref_slope, "k--", linewidth=1,
                  label=ref_label)

    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:.3g}" for h in hs])
    ax.minorticks_off()
    ax.set_xlabel("mesh size $h$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / filename
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.show()


if __name__ == "__main__":
    hs, results = run_study()

    results_dir.mkdir(parents=True, exist_ok=True)
    np.savez(results_dir / "h_sweep.npz", hs=hs, n_cells=n_cells_values,
             gamma_g=gamma_g_values,
             **{f"{k}_gg{gg}": results[gg][k]
                for gg in gamma_g_values for k in results[gg]})

    # Figure 1: coercivity side
    plot_vs_h(hs, results, "cond", "max", r"$\kappa(A_{uu})$",
              "h_sweep_cond.png", ref_slope=-2, ref_label=r"$h^{-2}$")

    # Figure 2: inf-sup constant
    plot_vs_h(hs, results, "beta", "min", r"inf-sup constant $\beta_h$",
              "h_sweep_beta.png", ref_slope=1, ref_label=r"$h^{1}$ (failing pair)")

    # Figure 3: pressure Schur pencil conditioning
    plot_vs_h(hs, results, "kappa_p", "max", r"$\kappa(BS^{-1}B^T + C,\ T)$",
              "h_sweep_kappa_p.png", ref_slope=-2,
              ref_label=r"$h^{-2}$ (for comparison)")