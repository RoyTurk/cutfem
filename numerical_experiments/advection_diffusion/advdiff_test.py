"""
advdiff_test.py
Single run of the advection-diffusion CutFEM solver to check that assembly,
solve and post-processing work end to end. Saves the result for ParaView.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from advdiff_problem import build_system, solve_system, save_vtk

# Geometry
xmin   = np.array([0.0, 0.0])
xmax   = np.array([1.0, 1.0])
center = np.array([0.2, 0.5])
radius = 0.1

# Physical parameters
mu = 5e-3
beta = np.array([1.0, 0.0])
g_disk = 1.0

# Stabilization parameters
gamma = 10.0        # Nitsche
gamma_mu = 0.1      # viscous ghost penalty
gamma_beta = 0.1    # CIP + convective ghost penalty

# Discretization
n_cells = 128
degree = 1

results_dir = Path("results")

print("=" * 65)
print(f"  n_cells={n_cells}, P{degree}, mu={mu}, beta=({beta[0]}, {beta[1]})")
print(f"  center=({center[0]}, {center[1]}), radius={radius}")
print(f"  gamma={gamma}, gamma_mu={gamma_mu}, gamma_beta={gamma_beta}")
print("=" * 65)

sd = build_system(
    n_cells=n_cells,
    mu=mu,
    beta=beta,
    gamma=gamma,
    gamma_mu=gamma_mu,
    gamma_beta=gamma_beta,
    center=center,
    radius=radius,
    xmin=xmin,
    xmax=xmax,
    g=g_disk,
    degree=degree,
)

sol = solve_system(sd)

pe = np.linalg.norm(beta) * sd.h / mu
n_dofs = sd.V.dofmap.index_map.size_local * sd.V.dofmap.index_map_bs

print(f"  h      = {sd.h:.5f}")
print(f"  Pe_T   = {pe:.2f}")
print(f"  n_dofs = {n_dofs}")
print(f"  uh     in [{sol.uh.x.array.min():.4f}, {sol.uh.x.array.max():.4f}] "
      f"(includes fictitious dofs)")

if not np.all(np.isfinite(sol.uh.x.array)):
    print("  WARNING: solution contains non-finite values")

# ParaView output
save_vtk(sd, sol, results_dir / "advdiff_test.pvd")