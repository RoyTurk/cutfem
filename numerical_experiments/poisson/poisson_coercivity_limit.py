"""
poisson_beta_sweep.py
Coercivity threshold study for the Poisson CutFEM solver.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import default_scalar_type as dtype

from poisson_problem import (
    build_system,
    solve_system,
    compute_condition,
    compute_eigenvalues,
)

# Parameters
radius  = 0.5
center  = np.array([0.0, 0.0], dtype=dtype)
xmin    = np.array([-1.0, -1.0], dtype=dtype)
xmax    = np.array([1.0, 1.0], dtype=dtype)
n_cells = 32
degree  = 1

beta_values = [2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]

tau_values = [0.0, 0.1, 1.0]

def run_sweep(tau):
    """Run the beta sweep for a single tau value."""
    label = f"tau={tau}"
    print(f"\n  --- {label} ---")
    print(f"  {'beta':>6} {'n_neg':>6} {'min_eig':>14} "
          f"{'cond':>14} {'L2':>12}")
    print(f"  {'-'*56}")

    records = []
    for beta in beta_values:
        sys  = build_system(
            n_cells=n_cells,
            beta=beta,
            tau=tau,
            center=center,
            radius=radius,
            xmin=xmin,
            xmax=xmax,
            degree=degree,
        )
        sol  = solve_system(sys)
        cond = compute_condition(sys)
        eigs = compute_eigenvalues(sys)

        min_eig = float(eigs[0]) if eigs is not None else np.nan
        n_neg   = int(np.sum(eigs < 0)) if eigs is not None else -1
        coercive = "OK" if n_neg == 0 else "FAIL"

        print(f"  {beta:>6} {n_neg:>6} {min_eig:>14.4e} "
              f"{cond:>14.4e} {sol.L2_u:>12.4e}  {coercive}")

        records.append({
            "beta":    beta,
            "n_neg":   n_neg,
            "min_eig": min_eig,
            "cond":    cond,
            "L2":      sol.L2_u,
        })

        sys.A.destroy()
        sys.b.destroy()
        del sys, sol, eigs

    return records


print("=" * 70)
print("  Poisson CutFEM — coercivity threshold study")
print(f"  n_cells={n_cells}, tau_values={tau_values}, radius={radius}, degree={degree}")
print("=" * 70)

all_results = {}
for tau_val in tau_values:
    all_results[tau_val] = run_sweep(tau_val)

print()
for tau_val in tau_values:
    records = all_results[tau_val]
    threshold = next(
        (r["beta"] for r in records if r["n_neg"] == 0), None)
    always = all(r["n_neg"] == 0 for r in records)
    if always:
        print(f"  tau={tau_val}: always coercive over tested range")
    elif threshold is not None:
        print(f"  tau={tau_val}: coercivity threshold  beta >= {threshold}")
    else:
        print(f"  tau={tau_val}: not coercive over tested range")

print("=" * 70)

markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, tau_val in enumerate(tau_values):
    records    = all_results[tau_val]
    betas      = [r["beta"]    for r in records]
    n_negs     = [r["n_neg"]   for r in records]
    min_eigs   = [r["min_eig"] for r in records]
    L2s        = [r["L2"]      for r in records]
    label      = rf"$\tau={tau_val}$"
    # colour comes from matplotlib's default cycle
    fmt        = f"{markers[i % len(markers)]}-"

    axes[0].semilogx(betas, n_negs,   fmt, color=f"C{i}",
                     linewidth=1.6, markersize=5, label=label)
    axes[1].semilogx(betas, min_eigs, fmt, color=f"C{i}",
                     linewidth=1.6, markersize=5, label=label)
    axes[2].loglog(  betas, L2s,      fmt, color=f"C{i}",
                     linewidth=1.6, markersize=5, label=label)

# Threshold line for no-ghost case (same colour as the tau=0.0 curve)
no_ghost_records = all_results[0.0]
threshold = next(
    (r["beta"] for r in no_ghost_records if r["n_neg"] == 0), None)
if threshold is not None:
    axes[0].axvline(threshold, color=f"C{tau_values.index(0.0)}",
                    linestyle=":", alpha=0.7,
                    label=rf"threshold $\beta={threshold}$")

axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_ylim(-2, None)
axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)

labels = [
    (r"number of negative eigenvalues", False),
    (r"smallest eigenvalue",     False),
    (r"$L^2$ error",             True),
]
for ax, (ylabel, _) in zip(axes, labels):
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / f"poisson_beta_sweep_n{n_cells}.png"
fig.savefig(plot_path, dpi=150)
print(f"\n  Saved {plot_path}")