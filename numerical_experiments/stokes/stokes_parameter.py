"""
stokes_parameter.py
Sensitivity study for the Stokes CutFEM solver
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stokes_problem import build_system, solve_system

# Parameters
n_cells = 32
center = np.array([0.0, 0.0])
radius = 0.5
mu = 1.0

gamma_mu_base = 10.0
gamma_g_base  = 0.1
gamma_p_base  = 0.0

# Sweep values for each parameter
gamma_mu_values = np.logspace(-2, 3, 20)   # 0.01 to 1000
gamma_g_values  = np.logspace(-4, 3, 20)   # 0.0001 to 1000
gamma_p_values  = np.logspace(-4, 3, 20)   # 0.0001 to 1000

# Helper
def run_one(gamma_mu, gamma_g, gamma_p):
    sys = build_system(
        n_cells  = n_cells,
        center   = center,
        gamma_mu = gamma_mu,
        gamma_g  = gamma_g,
        gamma_p  = gamma_p,
        radius   = radius,
        mu       = mu,
        v_degree = 2,
        p_degree = 1,
    )
    try:
        sol = solve_system(sys)
        return sol.L2_u, sol.L2_p
    except RuntimeError:
        return np.nan, np.nan

# Sweep gamma_mu  (gamma_g and gamma_p fixed at baseline)
print("=" * 60)
print(f"Sweeping gamma_mu  (gamma_g={gamma_g_base}, gamma_p={gamma_p_base})")
print(f"  {'gamma_mu':>12}  {'L2_u':>12}  {'L2_p':>12}")
print("-" * 42)

gmu_L2_u, gmu_L2_p = [], []
for gmu in gamma_mu_values:
    L2_u, L2_p = run_one(gmu, gamma_g_base, gamma_p_base)
    gmu_L2_u.append(L2_u)
    gmu_L2_p.append(L2_p)
    print(f"  {gmu:>12.4e}  {L2_u:>12.4e}  {L2_p:>12.4e}")

# Sweep gamma_g  (gamma_mu and gamma_p fixed at baseline)
print()
print("=" * 60)
print(f"Sweeping gamma_g  (gamma_mu={gamma_mu_base}, gamma_p={gamma_p_base})")
print(f"  {'gamma_g':>12}  {'L2_u':>12}  {'L2_p':>12}")
print("-" * 42)

gg_L2_u, gg_L2_p = [], []
for gg in gamma_g_values:
    L2_u, L2_p = run_one(gamma_mu_base, gg, gamma_p_base)
    gg_L2_u.append(L2_u)
    gg_L2_p.append(L2_p)
    print(f"  {gg:>12.4e}  {L2_u:>12.4e}  {L2_p:>12.4e}")

# Sweep gamma_p  (gamma_mu and gamma_g fixed at baseline)
print()
print("=" * 60)
print(f"Sweeping gamma_p  (gamma_mu={gamma_mu_base}, gamma_g={gamma_g_base})")
print(f"  {'gamma_p':>12}  {'L2_u':>12}  {'L2_p':>12}")
print("-" * 42)

gp_L2_u, gp_L2_p = [], []
for gp in gamma_p_values:
    L2_u, L2_p = run_one(gamma_mu_base, gamma_g_base, gp)
    gp_L2_u.append(L2_u)
    gp_L2_p.append(L2_p)
    print(f"  {gp:>12.4e}  {L2_u:>12.4e}  {L2_p:>12.4e}")

# Convert to arrays
gmu_L2_u = np.array(gmu_L2_u); gmu_L2_p = np.array(gmu_L2_p)
gg_L2_u  = np.array(gg_L2_u);  gg_L2_p  = np.array(gg_L2_p)
gp_L2_u  = np.array(gp_L2_u);  gp_L2_p  = np.array(gp_L2_p)

fig, ax = plt.subplots(figsize=(9, 6))

# x values in log10
log_gmu = np.log10(gamma_mu_values)
log_gg  = np.log10(gamma_g_values)
log_gp  = np.log10(gamma_p_values)

def safe_log10(arr):
    out = np.full_like(arr, np.nan)
    mask = arr > 0
    out[mask] = np.log10(arr[mask])
    return out

# One colour/marker per parameter: solid = velocity error, dashed = pressure error
sweeps = [
    (r"\gamma_\mu", log_gmu, gmu_L2_u, gmu_L2_p, "o"),
    (r"\gamma_g",   log_gg,  gg_L2_u,  gg_L2_p,  "s"),
    (r"\gamma_p",   log_gp,  gp_L2_u,  gp_L2_p,  "^"),
]
for i, (name, log_x, L2_u, L2_p, marker) in enumerate(sweeps):
    # colour comes from matplotlib's default cycle
    ax.plot(log_x, safe_log10(L2_u), f"{marker}-",
            color=f"C{i}", linewidth=1.6, markersize=5,
            label=rf"${name}$, velocity error")
    ax.plot(log_x, safe_log10(L2_p), f"{marker}--",
            color=f"C{i}", linewidth=1.6, markersize=5,
            label=rf"${name}$, pressure error")

ax.set_xlabel(r"$\log_{10}(\gamma)$", fontsize=12)
ax.set_ylabel(r"$\log_{10}(\mathrm{error})$", fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / "stokes_parameter.png"
fig.savefig(plot_path, dpi=150)
print(f"\nSaved {plot_path}")