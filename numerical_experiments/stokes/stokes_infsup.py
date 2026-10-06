"""
stokes_inf_sup.py
Inf-sup (pressure stability) study for the Stokes CutFEM solver.

Numerical inf-sup test of Chapelle & Bathe (1993, Prop. 2.2), adapted to the
stabilized Nitsche CutFEM of Burman & Hansbo (2014, Lemma 4.2, eq. 4.5):

    (B N^{-1} B^T + G_p) q = lambda M_T q,        q in Q_h / R

    B    matrix of b_h(q, v) = -(q, div v)_Omega + (q, v.n)_Gamma
    N    velocity norm matrix, ||v||_{V,T}^2 = mu ||grad v||^2_{Omega_T}
                                             + mu h^{-1} ||v||^2_Gamma
    G_p  pressure stabilization (ghost penalty + optional global CIP)
    M_T  pressure mass matrix on the active mesh Omega_T (weight 1/mu)

beta_h = sqrt(lambda_min) on the M_T-orthogonal complement of the constants.

N does not contain the velocity ghost penalty, so the result depends on
gamma_g only through G_p. gamma_g = 0 is the unstabilized test.

Study 1: sweep the disk center across one cell width at fixed h.
Study 2: mesh refinement, worst case over several shifts per h.

Serial only (dense generalized eigenproblem on the pressure space).
"""

from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

from stokes_problem import build_system

if MPI.COMM_WORLD.size > 1:
    raise RuntimeError("stokes_inf_sup.py must be run in serial.")

# Parameters
radius = 0.5
mu = 1.0
gamma_mu = 10
gamma_p = 0.0                            # no global pressure stabilization
gamma_g_values = [0.0, 0.01, 0.1, 1.0]   # ghost penalty (velocity and pressure)

xmin = np.array([-1.243, -1.243])
xmax = np.array([1.243, 1.243])
L = xmax[0] - xmin[0]

# Study 1: center sweep
n_cells_sweep = 32
n_sweep = 41
center_y = 0.0

# Study 2: refinement
n_cells_list = [16, 24, 32, 48, 64, 96]
n_shifts = 6                             # disk positions per mesh size

MAX_DENSE = 1.0e8                        # max entries of dense N^{-1} B^T
ZERO_TOL = 1e-10                         # relative threshold for "zero" eigenvalues

# Inf-sup computation

@dataclass
class InfSupData:
    beta:       float   # sqrt(lambda_min), constant removed
    lam_min:    float
    lam_max:    float
    kappa:      float   # lam_max / lam_min
    n_small:    int     # eigenvalues below ZERO_TOL * lam_max (spurious modes)
    rq_const:   float   # Rayleigh quotient of the constant pressure (should be ~0)
    skew:       float   # ||A_pu + A_up^T|| / ||A_up|| (should be ~0)
    n_dofs_Q:   int

def petsc_to_scipy(M):
    indptr, indices, data = M.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=M.getSize())

def compute_inf_sup(sys):
    """Solve (B N^{-1} B^T + G_p) q = lambda M_T q on Q_h / R."""
    B = petsc_to_scipy(sys.A_up).T.tocsr()          # (n_Q, n_V): b_h(q, v)
    A_pu = petsc_to_scipy(sys.A_pu)                  # -b_h(q, u)
    skew = spla.norm(A_pu + B) / spla.norm(B)

    N = petsc_to_scipy(sys.N_V).tocsc()
    G = petsc_to_scipy(sys.A_pp).toarray()
    M = petsc_to_scipy(sys.M_T).toarray()

    n_Q, n_V = B.shape
    if n_Q * n_V > MAX_DENSE:
        return None

    # sup_v (q^T B v)^2 / (v^T N v) = q^T B N^{-1} B^T q
    X = spla.splu(N).solve(B.T.toarray())            # N^{-1} B^T
    K = B @ X + G
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)

    # Constant pressure: expected exact kernel of b_h and of the ghost penalty
    ones = np.ones(n_Q)
    rq_const = float(ones @ K @ ones) / float(ones @ M @ ones)

    # Restrict to the M_T-orthogonal complement of the constants
    Z = sla.null_space((M @ ones)[None, :])
    lam = sla.eigh(Z.T @ K @ Z, Z.T @ M @ Z, eigvals_only=True)

    lam_min, lam_max = float(lam[0]), float(lam[-1])
    beta = np.sqrt(lam_min) if lam_min > 0 else 0.0
    kappa = lam_max / lam_min if lam_min > 0 else np.inf
    n_small = int(np.sum(lam < ZERO_TOL * lam_max))

    return InfSupData(beta, lam_min, lam_max, kappa, n_small,
                      rq_const, skew, n_Q)

def run(n_cells, center, gamma_g):
    sys = build_system(
        n_cells  = n_cells,
        center   = center,
        gamma_mu = gamma_mu,
        gamma_g  = gamma_g,
        gamma_p  = gamma_p,
        radius   = radius,
        xmin     = xmin,
        xmax     = xmax,
        mu       = mu,
        v_degree = 2,
        p_degree = 1,
    )
    return compute_inf_sup(sys)

def unpack(d):
    if d is None:
        return np.nan, np.nan, -1, np.nan, np.nan
    return d.beta, d.kappa, d.n_small, d.rq_const, d.skew

# Plotting

markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

def plot_curves(x, curves, xlabel, ylabel, filename, loglog=False, ref_slope=None):
    """
    One curve per gamma_g. Zero values are marked at the bottom (triangles),
    inf values at the top (crosses).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    plot = ax.loglog if loglog else ax.semilogy
    zeros, infs = [], []
    for i, (ggp, vals) in enumerate(curves.items()):
        vals = np.asarray(vals, dtype=float)
        ok = np.isfinite(vals) & (vals > 0)
        plot(x[ok], vals[ok], f"{markers[i % len(markers)]}-", color=f"C{i}",
             linewidth=1.6, markersize=5, label=rf"$\gamma_g={ggp}$")
        zeros.append((x[vals == 0], f"C{i}"))
        infs.append((x[np.isposinf(vals)], f"C{i}"))

    if ref_slope is not None:
        x_ref = np.array([x.min(), x.max()])
        y0 = np.nanmax([np.nanmax(np.where(np.asarray(v) > 0, v, np.nan))
                        for v in curves.values()])
        ax.loglog(x_ref, y0 * (x_ref / x_ref[1])**ref_slope, "k--",
                  linewidth=1, label=rf"slope {ref_slope} (reference)")

    y_bot, y_top = ax.get_ylim()
    for xs, c in zeros:
        if len(xs):
            ax.scatter(xs, np.full(len(xs), y_bot), color=c, marker="v",
                       s=60, zorder=5, clip_on=False)
    for xs, c in infs:
        if len(xs):
            ax.scatter(xs, np.full(len(xs), y_top), color=c, marker="x",
                       s=60, zorder=5, clip_on=False)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    plot_path = results_dir / filename
    fig.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    plt.show()

def print_header():
    print("=" * 75)
    print(f"  radius={radius}, mu={mu}, gamma_mu={gamma_mu}")
    print(f"  gamma_p={gamma_p}, gamma_g values: {gamma_g_values}")
    print("  Eigenproblem: (B N^-1 B^T + G_p) q = lambda M_T q on Q_h / R")
    print("=" * 75)

# Study 1: center sweep over one cell width

h_sweep = L / n_cells_sweep
shift_frac = np.linspace(0.0, 1.0, n_sweep)
center_x_values = shift_frac * h_sweep

print("\nStudy 1: center sweep")
print_header()
print(f"  n_cells={n_cells_sweep}, h={h_sweep:.4f}, "
      f"center_x in [0, h] ({n_sweep} points), center_y={center_y}")

sweep = {gg: {"beta": [], "kappa": [], "n_small": [], "rq": [], "skew": []}
         for gg in gamma_g_values}

for ggp in gamma_g_values:
    print(f"\ngamma_g = {ggp}")
    print(f"  {'s/h':>8} {'beta':>12} {'kappa_p':>12} {'n_small':>8}")
    print("  " + "-" * 44)
    for s, cx in zip(shift_frac, center_x_values):
        beta, kappa, n_small, rq, skew = unpack(
            run(n_cells_sweep, np.array([cx, center_y]), ggp))
        for key, val in zip(("beta", "kappa", "n_small", "rq", "skew"),
                            (beta, kappa, n_small, rq, skew)):
            sweep[ggp][key].append(val)
        print(f"  {s:>8.3f} {beta:>12.4e} {kappa:>12.4e} {n_small:>8d}")
    print(f"  max |RQ(constant)| = {np.nanmax(np.abs(sweep[ggp]['rq'])):.2e}, "
          f"max skew = {np.nanmax(sweep[ggp]['skew']):.2e}")

for ggp in gamma_g_values:
    for key in sweep[ggp]:
        sweep[ggp][key] = np.array(sweep[ggp][key])

plot_curves(shift_frac, {gg: sweep[gg]["beta"] for gg in gamma_g_values},
            r"disk center shift $s/h$", r"$\beta_h$", "infsup_sweep_beta.png")
plot_curves(shift_frac, {gg: sweep[gg]["kappa"] for gg in gamma_g_values},
            r"disk center shift $s/h$", r"$\lambda_{\max}/\lambda_{\min}$",
            "infsup_sweep_kappa.png")

# Study 2: refinement, worst case over shifts

print("\n\nStudy 2: refinement")
print_header()
print(f"  n_cells={n_cells_list}, {n_shifts} shifts per h, center_y={center_y}")

h_values = np.array([L / n for n in n_cells_list])
refine = {gg: {"beta_min": [], "kappa_max": [], "n_small_max": []}
          for gg in gamma_g_values}

for ggp in gamma_g_values:
    print(f"\ngamma_g = {ggp}")
    print(f"  {'n_cells':>8} {'h':>10} {'beta_min':>12} {'kappa_max':>12} {'n_small':>8}")
    print("  " + "-" * 56)
    for n, h in zip(n_cells_list, h_values):
        betas, kappas, smalls = [], [], []
        for k in range(n_shifts):
            cx = (k / n_shifts) * h
            beta, kappa, n_small, _, _ = unpack(
                run(n, np.array([cx, center_y]), ggp))
            betas.append(beta); kappas.append(kappa); smalls.append(n_small)
        beta_min = np.nanmin(betas) if np.any(np.isfinite(betas)) else np.nan
        kappa_max = np.nanmax(kappas) if np.any(~np.isnan(kappas)) else np.nan
        n_small_max = int(np.max(smalls))
        refine[ggp]["beta_min"].append(beta_min)
        refine[ggp]["kappa_max"].append(kappa_max)
        refine[ggp]["n_small_max"].append(n_small_max)
        print(f"  {n:>8d} {h:>10.4f} {beta_min:>12.4e} {kappa_max:>12.4e} "
              f"{n_small_max:>8d}")

for ggp in gamma_g_values:
    for key in refine[ggp]:
        refine[ggp][key] = np.array(refine[ggp][key])

plot_curves(h_values, {gg: refine[gg]["beta_min"] for gg in gamma_g_values},
            r"$h$", r"$\min_{\mathrm{shifts}} \beta_h$",
            "infsup_refine_beta.png", loglog=True, ref_slope=1)
plot_curves(h_values, {gg: refine[gg]["kappa_max"] for gg in gamma_g_values},
            r"$h$", r"$\max_{\mathrm{shifts}} \lambda_{\max}/\lambda_{\min}$",
            "infsup_refine_kappa.png", loglog=True)

# Save raw data
np.savez(results_dir / "infsup_results.npz",
         shift_frac=shift_frac, h_values=h_values,
         gamma_g_values=np.array(gamma_g_values),
         **{f"sweep_{key}_{gg}": sweep[gg][key]
            for gg in gamma_g_values for key in sweep[gg]},
         **{f"refine_{key}_{gg}": refine[gg][key]
            for gg in gamma_g_values for key in refine[gg]})
print(f"Saved {results_dir / 'infsup_results.npz'}")