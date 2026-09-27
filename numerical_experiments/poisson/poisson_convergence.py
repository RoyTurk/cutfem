"""
poisson_convergence.py
Mesh convergence study for the Poisson CutFEM solver.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from poisson_problem import build_system, solve_system

# Geometry
center = np.array([0.0, 0.0])
radius = 0.5
xmin   = np.array([-1.0, -1.0])
xmax   = np.array([1.0, 1.0])

# FE degree
degree = 1

# Mesh refinement levels
n_cells_list = [8, 16, 32, 64, 128, 256]

# Parameter configurations
configs = [
    dict(beta=10, tau=0.0),
    dict(beta=10, tau=1.0),
    dict(beta=0.0, tau=1.0)
]

def run_config(beta, tau):
    """Run the full refinement sweep for a single (beta, tau) pair."""
    print("=" * 65)
    print(f"  beta={beta}, tau={tau}")
    print(f"  expected: L2_u=O(h^2), H1_u=O(h^1)")
    print("=" * 65)
    print(f"  {'n_cells':>8} {'h':>10} {'n_dofs':>8} "
          f"{'L2_u':>12} {'H1_u':>12}")
    print("-" * 65)

    records = []
    for nc in n_cells_list:
        sys = build_system(
            n_cells=nc,
            beta=beta,
            tau=tau,
            center=center,
            radius=radius,
            xmin=xmin,
            xmax=xmax,
            degree=degree,
        )
        sol = solve_system(sys)

        n_dofs = sys.V.dofmap.index_map.size_local * sys.V.dofmap.index_map_bs
        print(f"  {nc:>8d} {sys.h:>10.5f} {n_dofs:>8d} "
              f"{sol.L2_u:>12.4e} {sol.H1_u:>12.4e}")

        records.append({
            "h":      sys.h,
            "n_dofs": n_dofs,
            "L2_u":   sol.L2_u,
            "H1_u":   sol.H1_u,
        })

        sys.A.destroy()
        sys.b.destroy()
        del sys, sol

    h_arr    = np.array([r["h"]    for r in records])
    L2_u_arr = np.array([r["L2_u"] for r in records])
    H1_u_arr = np.array([r["H1_u"] for r in records])

    def rates(err):
        return np.log(err[:-1] / err[1:]) / np.log(h_arr[:-1] / h_arr[1:])

    print("-" * 65)
    print("  Observed convergence rates:")
    print(f"  {'L2_u':>8}: " +
          "  ".join(f"{r:>6.2f}" for r in rates(L2_u_arr)) +
          "   (expected 2.00)")
    print(f"  {'H1_u':>8}: " +
          "  ".join(f"{r:>6.2f}" for r in rates(H1_u_arr)) +
          "   (expected 1.00)")
    print()

    return h_arr, L2_u_arr, H1_u_arr

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
    label = rf"$\beta={cfg['beta']},\ \tau={cfg['tau']}$"
    all_results[label] = run_config(**cfg)

fig, (ax_L2, ax_H1) = plt.subplots(1, 2, figsize=(11, 5))

for label, (h_arr, L2_u_arr, H1_u_arr) in all_results.items():
    ax_L2.loglog(h_arr, L2_u_arr, marker="o", markersize=5,
                 linewidth=1.6, label=label)
    ax_H1.loglog(h_arr, H1_u_arr, marker="o", markersize=5,
                 linewidth=1.6, label=label)

# Reference slopes (anchored to the last config)
last_h, last_L2, last_H1 = list(all_results.values())[-1]
add_slope(ax_L2, last_h, last_L2, exponent=2, label=r"$O(h^2)$")
add_slope(ax_H1, last_h, last_H1, exponent=1, label=r"$O(h^1)$")

for ax, ylabel, title in [
    (ax_L2, r"$\|u - u_h\|_{L^2(\Omega)}$", r"$L^2$ error"),
    (ax_H1, r"$|u - u_h|_{H^1(\Omega)}$",   r"$H^1$ seminorm error"),
]:
    ax.set_xlabel("mesh size $h$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / "poisson_convergence.png"
fig.savefig(plot_path, dpi=150)
print(f"Saved {plot_path}")
