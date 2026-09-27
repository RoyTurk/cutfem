"""
stokes_convergence.py
Mesh convergence study for the Stokes CutFEM solver.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stokes_problem import build_system, solve_system

# Geometry
center = np.array([0.0, 0.0])
radius = 0.5

# Physical and fixed parameters
mu      = 1.0
gamma_p = 0.0

# Mesh refinement levels
n_cells_list = [8, 16, 32, 64, 128, 256, 512]

# Parameter configurations
configs = [
    dict(gamma_mu=10, gamma_g=0.0),
    dict(gamma_mu=10, gamma_g=1.0),
    dict(gamma_mu=0.0, gamma_g=1.0),
]

def run_config(gamma_mu, gamma_g):
    """Run the full refinement sweep for a single (gamma_mu, gamma_g) pair."""
    print("=" * 75)
    print(f"  gamma_mu={gamma_mu}, gamma_g={gamma_g}, gamma_p={gamma_p}")
    print(f"  center=({center[0]}, {center[1]}), radius={radius}, mu={mu}")
    print(f"  elements: Q2/Q1 (quadrilateral mesh)")
    print(f"  expected: L2_u=O(h^3), H1_u=O(h^2), L2_p=O(h^2)")
    print("=" * 75)
    print(f"  {'n_cells':>8} {'h':>10} {'n_dofs':>8} "
          f"{'L2_u':>12} {'H1_u':>12} {'L2_p':>12} {'div_u':>12}")
    print("-" * 75)

    records = []
    for nc in n_cells_list:
        sys = build_system(
            n_cells=nc,
            center=center,
            gamma_mu=gamma_mu,
            gamma_g=gamma_g,
            gamma_p=gamma_p,
            radius=radius,
            mu=mu,
            v_degree=2,
            p_degree=1,
        )
        sol = solve_system(sys)

        n_dofs = sys.n_dofs_V + sys.n_dofs_Q
        print(f"  {nc:>8d} {sys.h:>10.5f} {n_dofs:>8d} "
              f"{sol.L2_u:>12.4e} {sol.H1_u:>12.4e} "
              f"{sol.L2_p:>12.4e} {sol.div_u:>12.4e}")

        records.append({
            "h":      sys.h,
            "n_dofs": n_dofs,
            "L2_u":   sol.L2_u,
            "H1_u":   sol.H1_u,
            "L2_p":   sol.L2_p,
            "div_u":  sol.div_u,
        })

        sys.A.destroy()
        sys.A_uu.destroy()
        sys.b.destroy()
        del sys, sol

    h_arr     = np.array([r["h"]     for r in records])
    L2_u_arr  = np.array([r["L2_u"]  for r in records])
    H1_u_arr  = np.array([r["H1_u"]  for r in records])
    L2_p_arr  = np.array([r["L2_p"]  for r in records])
    div_u_arr = np.array([r["div_u"] for r in records])

    def rates(err):
        return np.log(err[:-1] / err[1:]) / np.log(h_arr[:-1] / h_arr[1:])

    print("-" * 75)
    print("  Observed convergence rates:")
    print(f"  {'L2_u':>8}: " +
          "  ".join(f"{r:>6.2f}" for r in rates(L2_u_arr)) +
          "   (expected 3.00)")
    print(f"  {'H1_u':>8}: " +
          "  ".join(f"{r:>6.2f}" for r in rates(H1_u_arr)) +
          "   (expected 2.00)")
    print(f"  {'L2_p':>8}: " +
          "  ".join(f"{r:>6.2f}" for r in rates(L2_p_arr)) +
          "   (expected 2.00)")
    print()

    return h_arr, L2_u_arr, H1_u_arr, L2_p_arr, div_u_arr

def add_slope(ax, h_arr, err_arr, exponent, label, offset=0.3):
    """Dashed reference slope anchored just below the last data point."""
    h0       = h_arr[-1]
    h1       = h_arr[0]
    e_anchor = err_arr[-1] * offset
    e1       = e_anchor * (h1 / h0) ** exponent
    ax.plot([h1, h0], [e1, e_anchor],
            linestyle="--", linewidth=1.2, color="black", label=label)

all_results = {}
for cfg in configs:
    label = rf"$\gamma_\mu={cfg['gamma_mu']},\ \gamma_g={cfg['gamma_g']}$"
    all_results[label] = run_config(**cfg)

# (index into the result tuple, expected rate, ylabel, title)
panels = [
    (1, 3, r"$\|u - u_h\|_{L^2(\Omega)}$",          r"$L^2$ velocity error"),
    (2, 2, r"$|u - u_h|_{H^1(\Omega)}$",            r"$H^1$ velocity seminorm error"),
    (3, 2, r"$\|p - p_h\|_{L^2(\Omega)}$",          r"$L^2$ pressure error"),
    (4, 2, r"$\|\nabla \cdot u_h\|_{L^2(\Omega)}$", r"Incompressibility residual"),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 9))

for label, result in all_results.items():
    h_arr = result[0]
    for ax, (idx, _, _, _) in zip(axes.flat, panels):
        ax.loglog(h_arr, result[idx], marker="o", markersize=5,
                  linewidth=1.6, label=label)

# Reference slopes (anchored to the last config)
last_result = list(all_results.values())[-1]
for ax, (idx, exponent, ylabel, title) in zip(axes.flat, panels):
    add_slope(ax, last_result[0], last_result[idx], exponent=exponent,
              label=rf"$O(h^{exponent})$")
    ax.set_xlabel("mesh size $h$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / "stokes_convergence.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved {plot_path}")
