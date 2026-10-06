"""
stokes_cond_vs_h.py
Condition number vs mesh size study for the Stokes CutFEM solver.
"""

from pathlib import Path

from mpi4py import MPI

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from dolfinx import default_scalar_type as dtype

import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from stokes_problem import build_system, compute_condition

# Parameters
radius = 0.5
mu = 1.0
gamma_mu = 10.0
gamma_p = 0.0
center = np.array([0.0, 0.0])

xmin = np.array([-1.0, -1.0])
xmax = np.array([1.0, 1.0])

gamma_g_values = [0.0, 0.1, 1.0, 10.0]
n_cells_values = [8, 16, 32, 64]

# Same background domain as build_system defaults
# xmin = np.array([-1.0, -1.0], dtype=dtype)
# xmax = np.array([1.0, 1.0], dtype=dtype)

# Mesh check: show cut / full cells for each mesh before the sweep
def cell_bbox(mesh, cell_idx):
    conn   = mesh.topology.connectivity(mesh.topology.dim, 0)
    nodes  = conn.links(cell_idx)
    coords = mesh.geometry.x[nodes, :2]
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0)
    return x0, y0, x1 - x0, y1 - y0

def show_mesh(n_cells):
    impl_func = qugar.impl.create_disk(radius, center=center.astype(dtype))
    unf_mesh  = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, impl_func, n_cells, xmin, xmax,
        exclude_empty_cells=True, dtype=dtype,
    )
    unf_mesh.topology.create_connectivity(unf_mesh.topology.dim, 0)
    cut_cells  = unf_mesh.get_cut_cells()
    full_cells = unf_mesh.get_full_cells()

    fig, ax = plt.subplots(figsize=(6, 6))
    for c in full_cells:
        x0, y0, w, hc = cell_bbox(unf_mesh, c)
        ax.add_patch(patches.Rectangle(
            (x0, y0), w, hc,
            facecolor="#E1F5EE", edgecolor="#9FE1CB", linewidth=0.8,
        ))
    for c in cut_cells:
        x0, y0, w, hc = cell_bbox(unf_mesh, c)
        ax.add_patch(patches.Rectangle(
            (x0, y0), w, hc,
            facecolor="#EEEDFE", edgecolor="#AFA9EC", linewidth=0.8,
        ))

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            color="#534AB7", linewidth=1.5, linestyle="--")

    legend_elements = [
        Patch(facecolor="#EEEDFE", edgecolor="#AFA9EC", label=f"Cut cell ({len(cut_cells)})"),
        Patch(facecolor="#E1F5EE", edgecolor="#9FE1CB", label=f"Full cell ({len(full_cells)})"),
        Line2D([0], [0], color="#534AB7", linewidth=1.5, linestyle="--", label="Boundary"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
    ax.set_xlim(xmin[0] - 0.01, xmax[0] + 0.01)
    ax.set_ylim(xmin[1] - 0.01, xmax[1] + 0.01)
    ax.set_aspect("equal")
    ax.set_title(f"Mesh check — n_cells={n_cells},  h={(xmax[0] - xmin[0]) / n_cells:.4f}",
                 fontsize=10)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.tight_layout()
    plt.show()  # blocks until the window is closed

for n in n_cells_values:
    show_mesh(n)

# Sweep
print("=" * 75)
print("Stokes CutFEM — condition number vs mesh size")
print(f"  radius={radius}, center={center}, mu={mu}")
print(f"  gamma_mu={gamma_mu}, gamma_p={gamma_p}")
print(f"  gamma_g values: {gamma_g_values}")
print(f"  n_cells values: {n_cells_values}")
print(f"  Condition number: SVD of full A = [A_uu, A_up; A_pu, A_pp]"
      f" before pinning")
print("=" * 75)

# results[gamma_g] = {"h": [], "cond": []}
results = {gg: {"h": [], "cond": []} for gg in gamma_g_values}

for gg in gamma_g_values:
    print(f"\ngamma_g = {gg}")
    print(f"  {'n_cells':>8} {'h':>12} {'cond':>14}")
    print("  " + "-" * 36)
    for n in n_cells_values:

        sys = build_system(
            n_cells  = n,
            center   = center,
            gamma_mu = gamma_mu,
            gamma_g  = gg,
            gamma_p  = gamma_p,
            radius   = radius,
            mu       = mu,
            v_degree = 2,
            p_degree = 1,
        )

        cond = compute_condition(sys)

        results[gg]["h"].append(sys.h)
        results[gg]["cond"].append(cond)

        cond_str = f"{cond:.3e}" if np.isfinite(cond) else str(cond)
        print(f"  {n:>8d} {sys.h:>12.4e} {cond_str:>14}")

# Convert to arrays
for gg in gamma_g_values:
    for key in results[gg]:
        results[gg][key] = np.array(results[gg][key])

# Figure: condition number vs h
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

fig, ax = plt.subplots(figsize=(8, 5))
for i, gg in enumerate(gamma_g_values):
    h    = results[gg]["h"]
    cond = results[gg]["cond"]
    mask = np.isfinite(cond)
    # colour comes from matplotlib's default cycle
    ax.loglog(h[mask], cond[mask], f"{markers[i % len(markers)]}-",
              color=f"C{i}", linewidth=1.6, markersize=5,
              label=rf"$\gamma_g={gg}$")

# Dashed O(h^-2) reference slope, anchored just below the smallest
# condition number at the finest mesh
h_arr  = results[gamma_g_values[0]]["h"]
c_last = np.nanmin([results[gg]["cond"][-1] for gg in gamma_g_values])
if np.isfinite(c_last):
    c_anchor = c_last * 0.3
    c_first  = c_anchor * (h_arr[0] / h_arr[-1]) ** (-2)
    ax.plot([h_arr[0], h_arr[-1]], [c_first, c_anchor],
            linestyle="--", linewidth=1.2, color="k", label=r"$O(h^{-2})$")

ax.set_xlabel("mesh size $h$", fontsize=12)
ax.set_ylabel(r"$\kappa(\mathcal{A})$", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()

plt.show()

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
plot_path = results_dir / "cond_vs_h.png"
fig.savefig(plot_path, dpi=150)
print(f"\nSaved {plot_path}")
