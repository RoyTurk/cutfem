"""
test_navierstokes.py
First tests of the stationary Navier-Stokes CutFEM solver (DFG 2D-1, Re = 20).

    1. Geometry check: cut quadrature and orientation of the unfitted normal.
    2. Stokes solve (first Picard iterate only), saved for ParaView.
    3. Full Picard iteration, residual history, benchmark quantities,
       saved for ParaView.

Runs in serial or with MPI (mpirun -n N python test_navierstokes.py).
"""

from pathlib import Path

from mpi4py import MPI

import numpy as np
import ufl
import dolfinx.fem
from dolfinx import default_scalar_type as dtype

import qugar.dolfinx

from navierstokes_problem import (
    NSParams, build_system, update_parameters,
    picard_solve, save_vtk, postprocess,
)

OUT_DIR = Path(__file__).resolve().parent / "results" / "navierstokes"


def _print(*args, **kwargs):
    """Print on rank 0 only."""
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)


def _scalar(form, comm) -> float:
    """Assemble a scalar unfitted form and sum over all ranks."""
    f = qugar.dolfinx.form_custom(form)
    local = dolfinx.fem.assemble_scalar(f, coeffs=f.pack_coefficients())
    return float(comm.allreduce(local, op=MPI.SUM))


def _num_bc_dofs(sys) -> int:
    """Global number of DOFs constrained by the strong BCs (owned, summed over ranks)."""
    n_owned = 0
    for bc in sys.bcs:
        _, n_local = bc.dof_indices()
        n_owned += n_local
    return sys.unf_mesh.comm.allreduce(n_owned, op=MPI.SUM)


def print_system_info(sys):
    comm = sys.unf_mesh.comm
    n_cut = comm.allreduce(len(sys.unf_mesh.get_cut_cells()), op=MPI.SUM)
    nV = sys.V.dofmap.index_map.size_global * sys.V.dofmap.index_map_bs
    nQ = sys.Q.dofmap.index_map.size_global * sys.Q.dofmap.index_map_bs
    _print(f"h = {sys.h:.4e},  velocity DOFs = {nV},  pressure DOFs = {nQ},  "
           f"cut cells = {n_cut},  BC DOFs = {_num_bc_dofs(sys)},  "
           f"MPI ranks = {comm.size}")


def check_geometry(sys):
    """
    Sanity checks on the cut geometry:
      - fluid area      = |box| - pi r^2
      - cylinder length = 2 pi r
      - int (x - c) . n ds = -2 pi r^2 if n points OUT of the fluid
        (i.e. into the cylinder), +2 pi r^2 otherwise.
    """
    prm = sys.params
    mesh = sys.unf_mesh
    comm = mesh.comm
    one = dolfinx.fem.Constant(mesh, dtype(1.0))
    x = ufl.SpatialCoordinate(mesh)
    c = dolfinx.fem.Constant(mesh, np.array(prm.center, dtype=dtype))
    r = prm.radius

    box = (prm.xmax[0] - prm.xmin[0]) * (prm.xmax[1] - prm.xmin[1])
    area = _scalar(one * sys.dx, comm)
    length = _scalar(one * sys.ds_unf, comm)
    flux = _scalar(ufl.dot(x - c, sys.n_unf) * sys.ds_unf, comm)

    _print("Geometry check")
    _print(f"  fluid area      : {area:.8f}   (exact {box - np.pi * r**2:.8f})")
    _print(f"  cylinder length : {length:.8f}   (exact {2 * np.pi * r:.8f})")
    _print(f"  int (x-c).n ds  : {flux:.8f}   (expected {-2 * np.pi * r**2:.8f})")
    if flux > 0:
        _print("  WARNING: normal points INTO the fluid; Nitsche signs would be wrong.")


def main():
    # Replace gamma_u, gamma_p with your validated Stokes gamma_g
    prm = NSParams(
        gamma_u=0.1,
        gamma_p=0.1,
        gamma_N1=35.0,
        gamma_N2=35.0,
        ref=1,
    )
    tag = f"ref{prm.ref:g}"

    # ---------------- 1. Build + geometry ----------------
    sys = build_system(prm)

    from petsc4py import PETSc
    import dolfinx.fem.petsc
    from navierstokes_problem import _assemble_lhs, _assemble_rhs

    def has_nan(x):
        return bool(np.isnan(x).any() or np.isinf(x).any())

    sys = build_system(prm)

    # (a) single-form vs blocked assembly of the velocity block
    A00 = dolfinx.fem.petsc.assemble_matrix(sys.a_forms[0][0]); A00.assemble()
    print("single-form A_uu norm :", A00.norm())

    _assemble_lhs(sys)
    print("blocked A norm        :", sys.A.norm())

    # (b) RHS: plain blocked assembly, then lifting
    with sys.b.localForm() as loc: loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(sys.b, sys.L_forms)
    print("b after assemble, nan?:", has_nan(sys.b.array_r))
    _assemble_rhs(sys)
    print("b after lifting,  nan?:", has_nan(sys.b.array_r), " norm:", sys.b.norm())

    # (c) ghost-penalty pattern merged into A?
    print("nonzeros A:", sys.A.getInfo()["nz_used"], "  G:", sys.G.getInfo()["nz_used"])

    # (d) factorization
    ksp = PETSc.KSP().create(sys.unf_mesh.comm)
    ksp.setOperators(sys.A); ksp.setType("preonly")
    ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
    w = sys.A.createVecRight()
    ksp.solve(sys.b, w)
    print("KSP reason:", ksp.getConvergedReason(),
        "  PC failed reason:", ksp.getPC().getFailedReason())

    
    print_system_info(sys)
    check_geometry(sys)

    # ---------------- 2. Stokes solve only ----------------
    _print("\nStokes solve (first Picard iterate)")
    sys.params.max_iter = 0
    picard_solve(sys, verbose=False)
    save_vtk(sys, OUT_DIR / f"stokes_{tag}.pvd")

    # ---------------- 3. Full Picard iteration ----------------
    # Reuse the same system: reset the iterate (u_k = 0) instead of
    # rebuilding mesh, forms and matrices
    _print("\nPicard iteration (Navier-Stokes, Re = 20)")
    update_parameters(sys, reset=True)
    sys.params.max_iter = 50
    uh, ph, history = picard_solve(sys, verbose=True)
    results = postprocess(sys)
    save_vtk(sys, OUT_DIR / f"navierstokes_{tag}.pvd")

    if len(history) > 1:
        rates = np.array(history[1:]) / np.array(history[:-1])
        _print(f"\nIterations: {len(history)},  final ||r|| = {history[-1]:.3e},  "
               f"mean contraction factor = {rates.mean():.3f}")

    return results


if __name__ == "__main__":
    main()