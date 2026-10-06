"""
advdiff_peclet_sweep.py
Peclet sweep for the advection-diffusion CutFEM solver: profile of u_h along
the horizontal line y = y_line, from diffusion to advection dominated flow,
with and without CIP stabilization.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advdiff_problem import build_system, solve_system, sample_line

# Geometry
xmin   = np.array([0.0, 0.0])
xmax   = np.array([1.0, 1.0])
center = np.array([0.2, 0.5])
radius = 0.1
L      = xmax[0] - xmin[0]

# Physical parameters
beta = np.array([1.0, 0.0])
g_disk = 1.0
beta_norm = np.linalg.norm(beta)

# Stabilization parameters
gamma = 10.0        # Nitsche
gamma_mu = 0.1      # viscous ghost penalty

# Discretization
n_cells = 64
degree = 1
h = L / n_cells

# Grid Peclet numbers Pe_h = |beta| h / (2 mu), from diffusion (Pe_h < 1) to
# advection (Pe_h >> 1) dominated; mu is backed out for the fixed mesh
Pe_h_values = [0.1, 1, 10, 100]
mu_values = [beta_norm * h / (2 * Pe_h) for Pe_h in Pe_h_values]

# Configurations: no CIP vs CIP
gamma_beta_values = [0.0, 0.5]

# Sampling line
y_line = 0.50
n_samples = 1001
x_line = np.linspace(xmin[0], xmax[0], n_samples)

print("=" * 65)
print("Advection-diffusion CutFEM — Peclet sweep")
print(f"  n_cells={n_cells}, h={h:.5f}, P{degree}, beta=({beta[0]}, {beta[1]})")
print(f"  center=({center[0]}, {center[1]}), radius={radius}, g={g_disk}")
print(f"  gamma={gamma}, gamma_mu={gamma_mu}")
print(f"  gamma_beta values: {gamma_beta_values}")
print(f"  profile along y={y_line}")
print("=" * 65)

# results[gamma_beta][Pe_h] = u_h(x, y_line)
results = {gb: {} for gb in gamma_beta_values}

for gb in gamma_beta_values:
    print(f"\ngamma_beta = {gb}")
    print(f"  {'Pe_h':>8} {'Pe_L':>10} {'mu':>12} {'min u_h':>10} {'max u_h':>10}")
    print("  " + "-" * 54)
    for Pe_h, mu in zip(Pe_h_values, mu_values):
        sd = build_system(
            n_cells=n_cells,
            mu=mu,
            beta=beta,
            gamma=gamma,
            gamma_mu=gamma_mu,
            gamma_beta=gb,
            center=center,
            radius=radius,
            xmin=xmin,
            xmax=xmax,
            g=g_disk,
            degree=degree,
        )
        sol = solve_system(sd)

        u_line = sample_line(sd, sol, x_line, y_line)
        results[gb][Pe_h] = u_line

        Pe_L = beta_norm * L / mu
        print(f"  {Pe_h:>8.2f} {Pe_L:>10.1f} {mu:>12.3e} "
              f"{np.nanmin(u_line):>10.4f} {np.nanmax(u_line):>10.4f}")

        sd.A.destroy()
        sd.b.destroy()
        del sd, sol

# Plotting
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

# Pe_h is a magnitude: one sequential hue, light (diffusive) -> dark (advective)
colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(Pe_h_values)))

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)

for ax, gb in zip(axes, gamma_beta_values):
    for color, Pe_h, mu in zip(colors, Pe_h_values, mu_values):
        Pe_L = beta_norm * L / mu
        ax.plot(x_line, results[gb][Pe_h], color=color, linewidth=1.6,
                label=rf"$Pe_h={Pe_h:g}$ ($Pe_L={Pe_L:.0f}$)")

    # Disk cross-section on the sampling line
    half_chord = np.sqrt(max(radius ** 2 - (y_line - center[1]) ** 2, 0.0))
    if half_chord > 0:
        ax.axvspan(center[0] - half_chord, center[0] + half_chord,
                   color="0.85", zorder=0)

    title = r"no CIP ($\gamma_\beta=0$)" if gb == 0.0 else rf"CIP ($\gamma_\beta={gb}$)"
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(rf"$u_h(x, {y_line})$", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
axes[-1].set_xlabel("$x$", fontsize=12)
fig.tight_layout()

plt.show()

plot_path = results_dir / f"advdiff_peclet_sweep_n{n_cells}.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved {plot_path}")
