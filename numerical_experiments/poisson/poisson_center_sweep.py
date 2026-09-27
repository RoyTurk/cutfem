"""
poisson_center_sweep.py
Cut position study for the Poisson CutFEM solver.
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from dolfinx import default_scalar_type as dtype
from mpi4py import MPI

import qugar
import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from poisson_problem import (
    build_system,
    solve_system,
    compute_condition,
    compute_eigenvalues,
)


# Parameters
radius = 0.5
xmin = np.array([-1.0, -1.0], dtype=dtype)
xmax = np.array([1.0, 1.0], dtype=dtype)
n_cells = 32
degree = 1
beta = 10.0

h = (xmax[0] - xmin[0]) / n_cells

n_positions = 20
epsilon_values = np.linspace(0.0, 0.4 * h, n_positions)
eps_h = epsilon_values / h

tau_values = [0.0, 0.1, 1.0, 10.0]

def run_sweep(tau):
    """Run the cut position sweep for a single tau value."""
    label = f"tau={tau}"
    print(f"\n  --- {label} ---")
    print(f"  {'eps/h':>8} {'min_eig':>14} {'cond':>14} {'L2':>12}")
    print(f"  {'-'*52}")

    records = []
    for eps in epsilon_values:
        center = np.array([0.0 + eps, 0.0], dtype=dtype)

        sys = build_system(
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

        print(f"  {eps/h:>8.3f} {min_eig:>14.4e} {cond:>14.4e} {sol.L2_u:>12.4e}")

        records.append({
            "eps_h":   eps / h,
            "min_eig": min_eig,
            "cond":    cond,
            "L2":      sol.L2_u,
        })

        sys.A.destroy()
        sys.b.destroy()
        del sys, sol, eigs

    return records


print("=" * 65)
print("  Poisson CutFEM — cut position sweep")
print(f"  n_cells={n_cells}, beta={beta}, radius={radius}, degree={degree}")
print("=" * 65)

all_results = {}
for tau in tau_values:
    all_results[tau] = run_sweep(tau)

markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

panels = [
    ("min_eig", "smallest eigenvalue",  False),
    ("cond",    "condition number",      True),
    ("L2",      r"$L^2$ error",          True),
]

for ax, (key, ylabel, log_scale) in zip(axes, panels):
    for i, tau in enumerate(tau_values):
        records = all_results[tau]
        x_vals  = [r["eps_h"] for r in records]
        y_vals  = [r[key]     for r in records]
        plot_fn = ax.semilogy if log_scale else ax.plot
        # colour comes from matplotlib's default cycle
        plot_fn(x_vals, y_vals, f"{markers[i % len(markers)]}-",
                linewidth=1.6, markersize=5,
                label=rf"$\tau={tau}$")

    if not log_scale:
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\varepsilon / h$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
sweep_path = results_dir / f"poisson_center_sweep_n{n_cells}_beta{int(beta)}.png"
fig.savefig(sweep_path, dpi=150)
print(f"\n  Saved {sweep_path}")

# Find the eps giving the worst condition number without ghost penalty
cond_no_ghost = np.array([r["cond"] for r in all_results[0.0]])
worst_idx     = np.argmax(np.where(np.isfinite(cond_no_ghost), cond_no_ghost, 0))
worst_eps     = epsilon_values[worst_idx]
worst_eps_h   = eps_h[worst_idx]
worst_cond    = cond_no_ghost[worst_idx]

print(f"\n  Worst condition number (tau=0.0): {worst_cond:.4e}")
print(f"  at eps/h = {worst_eps_h:.3f}  (eps = {worst_eps:.6f})")

# Rebuild mesh at worst configuration
center_worst = np.array([0.0 + worst_eps, 0.0], dtype=dtype)
impl_worst   = qugar.impl.create_disk(radius, center=center_worst)
unf_worst    = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, impl_worst, n_cells, xmin, xmax,
    exclude_empty_cells=True, dtype=dtype,
)

cut_cells  = unf_worst.get_cut_cells()
full_cells = unf_worst.get_full_cells()
cut_cell_set = set(cut_cells.tolist())

# Build ghost facet set
tdim = unf_worst.topology.dim
fdim = tdim - 1
unf_worst.topology.create_connectivity(fdim, tdim)
unf_worst.topology.create_connectivity(tdim, 0)
unf_worst.topology.create_connectivity(fdim, 0)
f2c = unf_worst.topology.connectivity(fdim, tdim)

ghost_facet_ids = [
    f for f in range(f2c.num_nodes)
    if len(f2c.links(f)) == 2
    and cut_cell_set.intersection(f2c.links(f).tolist())
]

def cell_bbox(mesh, cell_idx):
    conn   = mesh.topology.connectivity(mesh.topology.dim, 0)
    nodes  = conn.links(cell_idx)
    coords = mesh.geometry.x[nodes, :2]
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    return x0, y0, x1 - x0, y1 - y0

def facet_endpoints(mesh, facet_idx):
    conn  = mesh.topology.connectivity(mesh.topology.dim - 1, 0)
    nodes = conn.links(facet_idx)
    return mesh.geometry.x[nodes[0], :2], mesh.geometry.x[nodes[1], :2]

fig2, ax = plt.subplots(figsize=(6, 6))

for c in full_cells:
    x0, y0, w, hc = cell_bbox(unf_worst, c)
    ax.add_patch(patches.Rectangle(
        (x0, y0), w, hc,
        facecolor="#E1F5EE", edgecolor="#9FE1CB", linewidth=0.8,
    ))

for c in cut_cells:
    x0, y0, w, hc = cell_bbox(unf_worst, c)
    ax.add_patch(patches.Rectangle(
        (x0, y0), w, hc,
        facecolor="#EEEDFE", edgecolor="#AFA9EC", linewidth=0.8,
    ))

for f in ghost_facet_ids:
    p0, p1 = facet_endpoints(unf_worst, f)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color="#D85A30", linewidth=1.0, solid_capstyle="round")

theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(
    center_worst[0] + radius * np.cos(theta),
    center_worst[1] + radius * np.sin(theta),
    color="#534AB7", linewidth=1.5, linestyle="--",
)

legend_elements = [
    Patch(facecolor="#EEEDFE", edgecolor="#AFA9EC", label="Cut cell"),
    Patch(facecolor="#E1F5EE", edgecolor="#9FE1CB", label="Full cell"),
    Line2D([0], [0], color="#D85A30", linewidth=2.5, label=r"Ghost facet $\mathcal{F}_h$"),
    Line2D([0], [0], color="#534AB7", linewidth=1.5, linestyle="--", label="Boundary"),
]
ax.legend(handles=legend_elements, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9, frameon=False)

# Zoom on the disk (with a margin of two cells) rather than the full domain
half_width = radius + 2 * h
ax.set_xlim(center_worst[0] - half_width, center_worst[0] + half_width)
ax.set_ylim(center_worst[1] - half_width, center_worst[1] + half_width)
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")

fig2.tight_layout()

plt.show()

worst_path = results_dir / f"poisson_worst_cut_n{n_cells}.png"
fig2.savefig(worst_path, dpi=150)
print(f"  Saved {worst_path}")