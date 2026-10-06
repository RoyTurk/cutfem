"""
advdiff_refinement.py
Mesh refinement study for the advection-diffusion CutFEM solver at the most
advection dominated global Peclet of advdiff_peclet_sweep.py: profile of u_h
along the horizontal line y = y_line, with and without CIP stabilization.
The coarsest and finest solutions are saved for ParaView.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advdiff_problem import build_system, solve_system, sample_line, save_vtk

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

# Diffusion fixed by the last case of advdiff_peclet_sweep.py:
# Pe_h = 20 on n_cells = 64, i.e. Pe_L = |beta| L / mu = 2560
n_cells_ref = 64
Pe_h_ref = 100.0
mu = 5e-3
Pe_L = beta_norm * L / mu

# Discretization
degree = 1
n_cells_list = [64, 128, 256, 512]

# Configurations: no CIP vs CIP
gamma_beta_values = [0.0, 0.5]

# Sampling line
y_line = 0.50
n_samples = 1001
x_line = np.linspace(xmin[0], xmax[0], n_samples)

results_dir = Path("results")

print("=" * 65)
print("Advection-diffusion CutFEM — mesh refinement study")
print(f"  mu={mu:.3e}, Pe_L={Pe_L:.0f}, P{degree}, beta=({beta[0]}, {beta[1]})")
print(f"  center=({center[0]}, {center[1]}), radius={radius}, g={g_disk}")
print(f"  gamma={gamma}, gamma_mu={gamma_mu}")
print(f"  gamma_beta values: {gamma_beta_values}")
print(f"  n_cells: {n_cells_list}")
print(f"  profile along y={y_line}")
print("=" * 65)

# results[gamma_beta][n_cells] = u_h(x, y_line)
results = {gb: {} for gb in gamma_beta_values}

for gb in gamma_beta_values:
    print(f"\ngamma_beta = {gb}")
    print(f"  {'n_cells':>8} {'h':>10} {'Pe_h':>8} {'n_dofs':>8} "
          f"{'min u_h':>10} {'max u_h':>10}")
    print("  " + "-" * 60)
    for nc in n_cells_list:
        sd = build_system(
            n_cells=nc,
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
        results[gb][nc] = u_line

        Pe_h = beta_norm * sd.h / (2 * mu)
        n_dofs = sd.V.dofmap.index_map.size_local * sd.V.dofmap.index_map_bs
        print(f"  {nc:>8d} {sd.h:>10.5f} {Pe_h:>8.2f} {n_dofs:>8d} "
              f"{np.nanmin(u_line):>10.4f} {np.nanmax(u_line):>10.4f}")

        # ParaView output for the coarsest and finest meshes
        if nc in (n_cells_list[0], n_cells_list[-1]):
            save_vtk(sd, sol, results_dir / f"advdiff_refinement_n{nc}_gb{gb}.pvd")

        sd.A.destroy()
        sd.b.destroy()
        del sd, sol

# Plotting
results_dir.mkdir(parents=True, exist_ok=True)

# n_cells is a magnitude: one sequential hue, light (coarse) -> dark (fine)
colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(n_cells_list)))

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)

for ax, gb in zip(axes, gamma_beta_values):
    for color, nc in zip(colors, n_cells_list):
        Pe_h = beta_norm * (L / nc) / (2 * mu)
        ax.plot(x_line, results[gb][nc], color=color, linewidth=1.6,
                label=rf"$n={nc}$ ($Pe_h={Pe_h:.3g}$)")

    # Disk cross-section on the sampling line
    half_chord = np.sqrt(max(radius ** 2 - (y_line - center[1]) ** 2, 0.0))
    if half_chord > 0:
        ax.axvspan(center[0] - half_chord, center[0] + half_chord,
                   color="0.85", zorder=0)

    title = r"no CIP ($\gamma_\beta=0$)" if gb == 0.0 else rf"CIP ($\gamma_\beta={gb}$)"
    ax.set_title(rf"{title}, $Pe_L={Pe_L:.0f}$", fontsize=11)
    ax.set_ylabel(rf"$u_h(x, {y_line})$", fontsize=12)
    ax.grid(True, alpha=0.3)

axes[0].legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
axes[-1].set_xlabel("$x$", fontsize=12)
fig.tight_layout()

plt.show()

plot_path = results_dir / f"advdiff_refinement_PeL{Pe_L:.0f}.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved {plot_path}")
