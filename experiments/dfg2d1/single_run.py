"""Single Navier-Stokes CutFEM run (DFG 2D-1, Re = 20).

1. Geometry check: cut quadrature and orientation of the unfitted normal.
2. Stokes solve (first Picard iterate), written for ParaView.
3. Full Picard iteration, benchmark quantities, written for ParaView.

Runs in serial or with MPI (mpirun -n N python single_run.py).
"""

import dolfinx.fem
import numpy as np
import ufl

from cutfem import core, study
from cutfem.problems import dfg2d1

PARAMS = dfg2d1.Params(n_y=40)


def check_geometry(s):
    """Fluid area, cylinder length and orientation of the normal on Gamma."""
    p, m = s.params, s.measures
    one = dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0))
    x = ufl.SpatialCoordinate(s.mesh)
    c = dolfinx.fem.Constant(s.mesh, np.array(p.center, dtype=core.DTYPE))
    r = p.radius
    box = (p.xmax[0] - p.xmin[0]) * (p.xmax[1] - p.xmin[1])

    flux = core.integrate(ufl.dot(x - c, m.n) * m.ds)
    study.print0("Geometry check")
    study.print0(f"  fluid area      : {core.integrate(one * m.dx):.8f}"
                 f"   (exact {box - np.pi * r ** 2:.8f})")
    study.print0(f"  cylinder length : {core.integrate(one * m.ds):.8f}"
                 f"   (exact {2 * np.pi * r:.8f})")
    study.print0(f"  int (x-c).n ds  : {flux:.8f}"
                 f"   (expected {-2 * np.pi * r ** 2:.8f})")
    if flux > 0:
        study.print0("  WARNING: the normal points into the fluid")


def compute():
    """Geometry check, Stokes solve, then the full Picard iteration."""
    out = study.RESULTS_DIR / "dfg2d1"
    tag = f"ny{PARAMS.n_y}"
    with dfg2d1.build(PARAMS) as system:
        study.print0(f"h = {system.h:.4e}, n_dofs = {system.n_dofs}, "
                     f"MPI ranks = {system.mesh.comm.size}")
        check_geometry(system)

        study.print0("\nStokes solve (first Picard iterate)")
        stokes = dfg2d1.solve(system, max_iter=0)
        dfg2d1.save_vtk(out / f"stokes_{tag}.pvd", system, stokes)

        study.print0("\nPicard iteration")
        dfg2d1.update_parameters(system, reset=True)
        solution = dfg2d1.solve(system, verbose=True)
        dfg2d1.save_vtk(out / f"navier_stokes_{tag}.pvd", system, solution)
        values = dfg2d1.benchmark(system, solution)

    study.print0(f"\n  {'quantity':<10}{'computed':>16}{'reference':>16}"
                 f"{'rel. error':>14}")
    for q, ref in dfg2d1.DFG_2D1_REF.items():
        study.print0(f"  {q:<10}{values[q]:>16.8f}{ref:>16.8f}"
                     f"{abs(values[q] - ref) / abs(ref):>14.3e}")
    return {"params": PARAMS, "history": solution[2], **values}


if __name__ == "__main__":
    study.run("dfg2d1/single_run", compute)
