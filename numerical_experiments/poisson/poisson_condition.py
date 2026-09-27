"""
poisson_condition_study.py
Condition number study for the Poisson CutFEM solver.
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
degree  = 1
beta    = 30.0  # known-coercive from beta sweep
tau     = 1.0  # ghost penalty prefactor

n_cells_list = [4, 8, 16, 32, 64]

tau_values = [0.0, tau]

def run_sweep(tau):
    """Run the mesh size sweep for a single tau value."""
    label = "with ghost" if tau != 0.0 else "no ghost  "
    print(f"\n  --- {label} ---")
    print(f"  {'n_cells':>8} {'h':>10} {'n_dofs':>8} "
          f"{'cond':>14} {'cond rate':>10} {'L2':>12}")
    print(f"  {'-'*65}")

    records   = []
    prev_cond = None
    prev_h    = None

    for nc in n_cells_list:
        sys  = build_system(
            n_cells=nc,
            beta=beta,
            tau=tau,
            center=center,
            radius=radius,
            xmin=xmin,
            xmax=xmax,
            degree=degree,
        )
        sol    = solve_system(sys)
        cond   = compute_condition(sys)
        n_dofs = sys.V.dofmap.index_map.size_local * sys.V.dofmap.index_map_bs

        if prev_cond is not None and np.isfinite(cond):
            cond_rate = np.log(cond / prev_cond) / np.log(sys.h / prev_h)
            rate_str  = f"{cond_rate:.2f}"
        else:
            rate_str = "  —"

        print(f"  {nc:>8d} {sys.h:>10.4f} {n_dofs:>8d} "
              f"{cond:>14.4e} {rate_str:>10} {sol.L2_u:>12.4e}")

        records.append({
            "h":      sys.h,
            "n_dofs": n_dofs,
            "cond":   cond,
            "L2":     sol.L2_u,
        })

        if np.isfinite(cond):
            prev_cond = cond
            prev_h    = sys.h

        sys.A.destroy()
        sys.b.destroy()
        del sys, sol

    return records


print("=" * 70)
print("  Poisson CutFEM — condition number study")
print(f"  beta={beta}, tau={tau}, radius={radius}, degree={degree}")
print(f"  Theory: cond(K_h) <= C * h^{{-2}}  (Theorem 3.3)")
print("=" * 70)

all_results = {}
for tau_val in tau_values:
    all_results[tau_val] = run_sweep(tau_val)

print("\n" + "=" * 70)
print("  Expected asymptotic condition number rate: -2.00")
print("=" * 70)

# ── Plot ──────────────────────────────────────────────────────────────────────

markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

fig, (ax_cond, ax_L2) = plt.subplots(1, 2, figsize=(13, 5))

for i, tau_val in enumerate(tau_values):
    records  = all_results[tau_val]
    h_arr    = np.array([r["h"]    for r in records])
    cond_arr = np.array([r["cond"] for r in records])
    L2_arr   = np.array([r["L2"]   for r in records])
    label    = "no ghost" if tau_val == 0.0 else rf"ghost ($\tau={tau_val}$)"
    # colour comes from matplotlib's default cycle
    fmt      = f"{markers[i % len(markers)]}-"

    valid = np.isfinite(cond_arr) & (cond_arr > 0)
    ax_cond.loglog(h_arr[valid], cond_arr[valid], fmt,
                   color=f"C{i}", linewidth=1.6, markersize=5, label=label)
    ax_L2.loglog(h_arr, L2_arr, fmt,
                 color=f"C{i}", linewidth=1.6, markersize=5, label=label)

# Reference slopes anchored to the ghost penalty curve
ghost_records = all_results[tau]
h_ref    = np.array([r["h"]    for r in ghost_records])
cond_ref = np.array([r["cond"] for r in ghost_records])
L2_ref   = np.array([r["L2"]   for r in ghost_records])

valid = np.isfinite(cond_ref) & (cond_ref > 0)
if valid.any():
    ax_cond.loglog(
        h_ref[valid],
        cond_ref[valid][0] * (h_ref[valid] / h_ref[valid][0]) ** (-2),
        "k--", alpha=0.5, label=r"$O(h^{-2})$",
    )

ax_L2.loglog(
    h_ref,
    L2_ref[0] * (h_ref / h_ref[0]) ** 2,
    "k--", alpha=0.5, label=r"$O(h^2)$",
)

for ax, xlabel, ylabel, title in [
    (ax_cond, "mesh size $h$", "condition number", "Condition number vs mesh size"),
    (ax_L2,   "mesh size $h$", r"$L^2$ error",     r"$L^2$ convergence (accuracy check)"),
]:
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title,   fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / f"poisson_condition_study_beta{int(beta)}.png"
fig.savefig(plot_path, dpi=150)
print(f"\n  Saved {plot_path}")