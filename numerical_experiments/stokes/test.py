"""
stokes_center_sweep.py
Center sweep study for the Stokes CutFEM solver.

For every disk position and every gamma_g, the SAME assembled system is used for
    - coercivity side : condition number of the velocity block A_uu
    - inf-sup side    : inf-sup constant beta and Schur-pencil condition kappa_p
    - accuracy        : L2 errors of velocity and pressure
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stokes_problem import build_system, solve_system, compute_condition
from infsup_test import compute_inf_sup

# Parameters
radius = 0.5
mu = 1.0
gamma_mu = 10
gamma_p = 0.0
n_cells = 32

xmin = np.array([-1.243, -1.243])
xmax = np.array([1.243, 1.243])

gamma_g_values = [0.0, 0.1, 1.0]

n_sweep = 41
center_x_values = np.linspace(-0.3, 0.3, n_sweep)
center_y = 0.0

# Sweep
print("=" * 75)
print("Stokes CutFEM — center sweep study")
print(f"  radius={radius}, n_cells={n_cells}, mu={mu}")
print(f"  gamma_mu={gamma_mu}, gamma_p={gamma_p}")
print(f"  gamma_g values: {gamma_g_values}")
print(f"  center_x in [{center_x_values[0]:.3f}, {center_x_values[-1]:.3f}]"
      f" ({n_sweep} points), center_y={center_y}")
print("  cond    : SVD condition number of the velocity block A_uu")
print("  beta    : inf-sup constant, lambda_min(B S^-1 B^T + C, T)^(1/2)")
print("  kappa_p : lambda_max/lambda_min of the same pressure Schur pencil")
print("=" * 75)

keys = ["cond", "beta", "kappa_p", "n_zero", "L2_u", "L2_p"]
results = {gg: {k: [] for k in keys} for gg in gamma_g_values}

for gg in gamma_g_values:
    print(f"\ngamma_g = {gg}")
    print(f"  {'center_x':>10} {'cond':>11} {'beta':>11} {'kappa_p':>11}"
          f" {'nz':>3} {'L2_u':>11} {'L2_p':>11}")
    print("  " + "-" * 75)
    for cx in center_x_values:

        sys = build_system(
            n_cells  = n_cells,
            center   = np.array([cx, center_y]),
            gamma_mu = gamma_mu,
            gamma_g  = gg,
            gamma_p  = gamma_p,
            radius   = radius,
            xmin     = xmin,
            xmax     = xmax,
            mu       = mu,
            v_degree = 2,
            p_degree = 1,
        )

        # Coercivity side and inf-sup side, both on the unpinned system
        cond = compute_condition(sys)
        isup = compute_inf_sup(sys, mu=mu)        # must come before solve_system

        try:
            sol = solve_system(sys)
            L2_u = sol.L2_u
            L2_p = sol.L2_p
        except RuntimeError as e:
            print(f"  {cx:>+10.4f}  [solve failed: {e}]")
            L2_u = np.nan
            L2_p = np.nan

        vals = dict(cond=cond, beta=isup["beta"], kappa_p=isup["kappa_p"],
                    n_zero=isup["n_zero"], L2_u=L2_u, L2_p=L2_p)
        for k in keys:
            results[gg][k].append(vals[k])

        def fmt(x):
            return f"{x:.3e}" if np.isfinite(x) else ("inf" if x == np.inf else "nan")
        print(f"  {cx:>+10.4f} {fmt(cond):>11} {fmt(isup['beta']):>11}"
              f" {fmt(isup['kappa_p']):>11} {isup['n_zero']:>3d}"
              f" {fmt(L2_u):>11} {fmt(L2_p):>11}")

# Convert to arrays
for gg in gamma_g_values:
    for key in results[gg]:
        results[gg][key] = np.array(results[gg][key], dtype=float)

# Plotting
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

def plot_sweep(key, ylabel, filename, mark_inf=False, mark_zero=False):
    """One curve per gamma_g of results[gg][key] vs disk center x-position.
    mark_inf : values == inf drawn as 'x' at the top   (singular system)
    mark_zero: values <= 0   drawn as 'x' at the bottom (zero inf-sup constant)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    top_points, bottom_points = [], []
    for i, gg in enumerate(gamma_g_values):
        vals = results[gg][key]
        mask = np.isfinite(vals) & (vals > 0)
        ax.semilogy(center_x_values[mask], vals[mask],
                    f"{markers[i % len(markers)]}-", color=f"C{i}",
                    linewidth=1.6, markersize=5, label=rf"$\gamma_g={gg}$")
        if mark_inf and (vals == np.inf).any():
            top_points.append((center_x_values[vals == np.inf], f"C{i}"))
        if mark_zero and (vals <= 0).any():
            bottom_points.append((center_x_values[vals <= 0], f"C{i}"))

    y_bot, y_top = ax.get_ylim()
    for x_pts, color in top_points:
        ax.scatter(x_pts, np.full(len(x_pts), y_top), color=color,
                   marker="x", s=60, zorder=5)
    for x_pts, color in bottom_points:
        ax.scatter(x_pts, np.full(len(x_pts), y_bot), color=color,
                   marker="x", s=60, zorder=5)

    ax.set_xlabel("disk center $x$-position", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plot_path = results_dir / filename
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.show()

# Figure 1: Condition number of the velocity block (coercivity side)
plot_sweep("cond", r"$\kappa(A_{uu})$", "sweep_cond.png", mark_inf=True)

# Figure 2: Inf-sup constant (inf-sup side)
plot_sweep("beta", r"inf-sup constant $\beta_h$", "sweep_beta.png", mark_zero=True)

# Figure 3: Condition number of the pressure Schur pencil (inf-sup side)
plot_sweep("kappa_p", r"$\kappa(BS^{-1}B^T + C,\ T)$", "sweep_kappa_p.png",
           mark_inf=True)

# Figure 4: L2 velocity error
plot_sweep("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$", "sweep_L2_u.png")

# Figure 5: L2 pressure error
plot_sweep("L2_p", r"$\|p - p_h\|_{L^2(\Omega)}$", "sweep_L2_p.png")