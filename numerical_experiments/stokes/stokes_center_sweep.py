"""
stokes_center_sweep.py
Center sweep study for the Stokes CutFEM solver.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stokes_problem import build_system, solve_system, compute_condition

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
print(f"  Condition number: SVD of full A = [A_uu, A_up; A_pu, A_pp]"
      f" before pinning")
print("=" * 75)

# results[gamma_g] = {"cond": [], "L2_u": [], "L2_p": []}
results = {gg: {"cond": [], "L2_u": [], "L2_p": []} for gg in gamma_g_values}

for gg in gamma_g_values:
    print(f"\ngamma_g = {gg}")
    print(f"  {'center_x':>10} {'cond':>14} {'L2_u':>12} {'L2_p':>12}")
    print("  " + "-" * 52)
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

        cond = compute_condition(sys)

        try:
            sol = solve_system(sys)
            L2_u = sol.L2_u
            L2_p = sol.L2_p
        except RuntimeError as e:
            print(f"  {cx:>+10.4f}  [solve failed: {e}]")
            L2_u = np.nan
            L2_p = np.nan

        results[gg]["cond"].append(cond)
        results[gg]["L2_u"].append(L2_u)
        results[gg]["L2_p"].append(L2_p)

        cond_str = f"{cond:.3e}" if np.isfinite(cond) else "inf"
        L2_u_str = f"{L2_u:.3e}" if np.isfinite(L2_u) else "FAILED"
        L2_p_str = f"{L2_p:.3e}" if np.isfinite(L2_p) else "FAILED"
        print(f"  {cx:>+10.4f} {cond_str:>14} {L2_u_str:>12} {L2_p_str:>12}")

# Convert to arrays
for gg in gamma_g_values:
    for key in results[gg]:
        results[gg][key] = np.array(results[gg][key])

# Plotting
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

def plot_sweep(key, ylabel, filename, mark_inf=False):
    """One curve per gamma_g of results[gg][key] vs disk center x-position."""
    fig, ax = plt.subplots(figsize=(8, 5))
    inf_points = []
    for i, gg in enumerate(gamma_g_values):
        vals = results[gg][key]
        mask = np.isfinite(vals)
        # colour comes from matplotlib's default cycle
        ax.semilogy(center_x_values[mask], vals[mask],
                    f"{markers[i % len(markers)]}-", color=f"C{i}",
                    linewidth=1.6, markersize=5, label=rf"$\gamma_g={gg}$")
        if mark_inf:
            inf_mask = vals == np.inf
            if inf_mask.any():
                inf_points.append((center_x_values[inf_mask], f"C{i}"))

    # Mark inf values (singular system) at the top of the plot, once all
    # finite curves are drawn so the y-limits are final
    y_top = ax.get_ylim()[1]
    for x_inf, color in inf_points:
        ax.scatter(x_inf, np.full(len(x_inf), y_top),
                   color=color, marker="x", s=60, zorder=5)

    ax.set_xlabel("disk center $x$-position", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plt.show()

    plot_path = results_dir / filename
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")

# Figure 1: Condition number
plot_sweep("cond", r"$\kappa(\mathcal{A})$", "sweep_cond.png", mark_inf=True)

# Figure 2: L2 velocity error
plot_sweep("L2_u", r"$\|u - u_h\|_{L^2(\Omega)}$", "sweep_L2_u.png")

# Figure 3: L2 pressure error
plot_sweep("L2_p", r"$\|p - p_h\|_{L^2(\Omega)}$", "sweep_L2_p.png")