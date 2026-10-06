"""
stokes_test.py
Single run of the Stokes CutFEM solver to check that assembly, solve and
post-processing work end to end. Saves the result for ParaView.
"""

from pathlib import Path

import numpy as np

from stokes_problem import build_system, solve_system, save_vtk

# Geometry
xmin   = np.array([-1.0, -1.0])
xmax   = np.array([1.0, 1.0])
center = np.array([0.0, 0.0])
radius = 0.5

# Physical parameters
mu = 1.0

# Stabilization parameters
gamma_mu = 10.0
gamma_g  = 0.1
gamma_p  = 1.0

# Discretization
n_cells  = 64
v_degree = 1
p_degree = 1

results_dir = Path("results")

print("=" * 65)
print(f"  n_cells={n_cells}, P{v_degree}-P{p_degree}, mu={mu}")
print(f"  center=({center[0]}, {center[1]}), radius={radius}")
print(f"  gamma_mu={gamma_mu}, gamma_g={gamma_g}, gamma_p={gamma_p}")
print("=" * 65)

sys = build_system(
    n_cells=n_cells,
    center=center,
    gamma_mu=gamma_mu,
    gamma_g=gamma_g,
    gamma_p=gamma_p,
    radius=radius,
    xmin=xmin,
    xmax=xmax,
    mu=mu,
    v_degree=v_degree,
    p_degree=p_degree,
)
sol = solve_system(sys)

print(f"  h      = {sys.h:.5f}")
print(f"  n_dofs = {sys.n_dofs_V + sys.n_dofs_Q} "
      f"(velocity {sys.n_dofs_V}, pressure {sys.n_dofs_Q})")
print(f"  L2_u   = {sol.L2_u:.4e}")
print(f"  H1_u   = {sol.H1_u:.4e}")
print(f"  L2_p   = {sol.L2_p:.4e}")
print(f"  div_u  = {sol.div_u:.4e}")

if not (np.all(np.isfinite(sol.uh.x.array))
        and np.all(np.isfinite(sol.ph.x.array))):
    print("  WARNING: solution contains non-finite values")

# ParaView output
save_vtk(sys, sol, results_dir / "stokes_test.pvd")
