"""
convergence_navierstokes.py
Mesh convergence study for the stationary Navier-Stokes CutFEM solver
(DFG benchmark 2D-1, Re = 20): c_D, c_L and dp for several refinement levels.

Usage:
    python convergence_navierstokes.py                      # default levels
    python convergence_navierstokes.py --refs 0.25 0.5 1 2  # custom levels
    mpirun -n 4 python convergence_navierstokes.py --refs 1 2 4

Output (in results/navierstokes/convergence/):
    convergence.csv   - h, DOFs, Picard iterations, values, relative errors
    convergence.png   - values vs h and relative errors vs h (log-log)
"""

import argparse
import csv
import time
from pathlib import Path

from mpi4py import MPI

import numpy as np

from navierstokes_problem import (
    DFG_2D1_REF, NSParams, build_system, picard_solve, postprocess,
)

OUT_DIR = Path(__file__).resolve().parent / "results" / "navierstokes" / "convergence"
QUANTITIES = ("c_D", "c_L", "dp")
LABELS = {"c_D": r"$c_D$", "c_L": r"$c_L$", "dp": r"$\Delta p$"}


def _print(*args, **kwargs):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs, flush=True)


def run_level(ref: float, base: dict) -> dict:
    """Build, solve and post-process one refinement level."""
    prm = NSParams(ref=ref, **base)
    t0 = time.perf_counter()

    sys = build_system(prm)
    _, _, history = picard_solve(sys, verbose=False)
    res = postprocess(sys, verbose=False)

    n_dofs = (sys.V.dofmap.index_map.size_global * sys.V.dofmap.index_map_bs
              + sys.Q.dofmap.index_map.size_global)
    row = {
        "ref": ref,
        "h": sys.h,
        "dofs": n_dofs,
        "picard": len(history),
        "time": time.perf_counter() - t0,
        **res,
    }
    for q in QUANTITIES:
        row[f"err_{q}"] = abs(res[q] - DFG_2D1_REF[q]) / abs(DFG_2D1_REF[q])

    _print(f"ref = {ref:<5g} h = {sys.h:.3e}  DOFs = {n_dofs:>9d}  "
           f"Picard = {len(history):>2d}  time = {row['time']:7.1f} s  |  "
           + "  ".join(f"{q} = {res[q]:.8f}" for q in QUANTITIES))
    return row


def observed_orders(rows: list, key: str) -> list:
    """Observed order between successive levels: log(e_i / e_{i+1}) / log(h_i / h_{i+1})."""
    orders = [np.nan]
    for a, b in zip(rows[:-1], rows[1:]):
        ea, eb = a[f"err_{key}"], b[f"err_{key}"]
        if ea > 0 and eb > 0:
            orders.append(np.log(ea / eb) / np.log(a["h"] / b["h"]))
        else:
            orders.append(np.nan)
    return orders


def print_summary(rows: list) -> None:
    _print("\nRelative errors and observed orders")
    header = f"  {'h':>10}{'DOFs':>10}"
    for q in QUANTITIES:
        header += f"{'err ' + q:>14}{'order':>7}"
    _print(header)

    orders = {q: observed_orders(rows, q) for q in QUANTITIES}
    for i, row in enumerate(rows):
        line = f"  {row['h']:>10.3e}{row['dofs']:>10d}"
        for q in QUANTITIES:
            o = orders[q][i]
            line += f"{row[f'err_{q}']:>14.3e}" + (f"{o:>7.2f}" if np.isfinite(o) else f"{'-':>7}")
        _print(line)


def save_csv(rows: list, path: Path) -> None:
    keys = ["ref", "h", "dofs", "picard", "time",
            *QUANTITIES, *(f"err_{q}" for q in QUANTITIES)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def plot(rows: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = np.array([r["h"] for r in rows])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)

    for j, q in enumerate(QUANTITIES):
        vals = np.array([r[q] for r in rows])
        errs = np.array([r[f"err_{q}"] for r in rows])
        ref = DFG_2D1_REF[q]

        # Top row: computed values vs h, with the reference value
        ax = axes[0, j]
        ax.semilogx(h, vals, "o-", label="CutFEM")
        ax.axhline(ref, color="k", ls="--", lw=1, label=f"reference {ref:.6g}")
        ax.set_xlabel(r"$h$")
        ax.set_title(LABELS[q])
        ax.invert_xaxis()                      # refinement goes to the right
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

        # Bottom row: relative error vs h (log-log) with reference slopes
        ax = axes[1, j]
        mask = errs > 0
        ax.loglog(h[mask], errs[mask], "o-", label="relative error")
        if mask.sum() >= 1:
            # anchor guide lines at the finest point instead of the coarsest
            h0, e0 = h[mask][-1], errs[mask][-1]
            for p, style in ((2, ":"), (3, "-.")):
                ax.loglog(h, e0 * (h / h0) ** p, "k" + style, lw=1,
                          label=rf"$O(h^{p})$")
        ax.set_xlabel(r"$h$")
        ax.set_xticks(h)
        ax.set_xticklabels([f"{x:.3g}" for x in h], rotation=45)
        ax.minorticks_off()
        ax.set_ylabel("relative error")
        ax.invert_xaxis()
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("DFG 2D-1 (Re = 20): mesh convergence of the CutFEM solver")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refs", type=float, nargs="+",
                        default=[0.25, 0.5, 1.0, 2.0],
                        help="refinement levels (h ~ 0.01 / ref)")
    args = parser.parse_args()

    # Same parameters on every level (replace gamma_u, gamma_p with your Stokes values)
    base = dict(gamma_u=0.1, gamma_p=0.1, gamma_N1=35.0, gamma_N2=35.0)

    _print(f"Convergence study, levels: {args.refs}\n")
    rows = [run_level(ref, base) for ref in sorted(args.refs)]

    print_summary(rows)

    if MPI.COMM_WORLD.rank == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        save_csv(rows, OUT_DIR / "convergence.csv")
        plot(rows, OUT_DIR / "convergence.png")
        print(f"\nSaved: {OUT_DIR / 'convergence.csv'}")
        print(f"Saved: {OUT_DIR / 'convergence.png'}")


if __name__ == "__main__":
    main()