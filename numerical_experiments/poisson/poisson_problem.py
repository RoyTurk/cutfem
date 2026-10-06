"""
poisson_problem.py
Core module for the Poisson equation solver
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np
import ufl
from dolfinx import default_scalar_type as dtype
from petsc4py import PETSc

import qugar
import qugar.impl
import qugar.dolfinx
from qugar.dolfinx import dsu, UnfittedNormal
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cutfem_utils import ghost_facet_measure


@dataclass
class SystemData:
    # PETSc objects
    A:          object
    b:          object
    # FE space and mesh
    V:          object
    unf_mesh:   object
    # Mesh parameter
    h:          float
    # Exact solution
    u_ex:       object
    # Integration measure over physical domain
    dx:         object


@dataclass
class SolveData:
    uh:     object  # Discrete solution
    L2_u:   float   # L2 error
    H1_u:   float   # H1 seminorm error


def exact_solution(x):
    """
    Returns u_ex as a UFL expression on the given spatial coordinates.
    u = sin(pi * x) * sin(pi * y)
    """
    return ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])


def build_system(
        n_cells:    int,
        beta:       float,
        tau:        float,
        center:     np.ndarray,
        radius:     float=0.5,
        xmin:       np.ndarray=None,
        xmax:       np.ndarray=None,
        degree:     int=1,
) -> SystemData:
    """
    Build and assemble the full Poisson CutFEM system.

    Parameters
    ------------
    n_cells:    number of cells per side on the background mesh
    beta:       Nitsche penalty parameter
    tau:        Ghost stabilization parameter
    center:     (2,) array, center of immersed disk
    radius:     Immersed disk radius
    xmin:       lower corner of background mesh
    xmax:       upper corner of background domain
    degree:     polynomial degree of FE space

    Returns
    ------------
    SystemData  all assemble objects needed to solve the system
    """
    if xmin is None:
        xmin = np.array([-1.0, -1.0], dtype=dtype)
    if xmax is None:
        xmax = np.array([1.0, 1.0], dtype=dtype)

    center = np.asarray(center, dtype=dtype)
    h = (xmax[0] - xmin[0]) / n_cells

    # Mesh
    impl_func = qugar.impl.create_disk(radius, center=center)
    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, impl_func, n_cells, xmin, xmax,
        exclude_empty_cells=True, dtype=dtype
    )

    # Exact solution and source term
    x = ufl.SpatialCoordinate(unf_mesh)
    u_ex = exact_solution(x)
    f = - ufl.div(ufl.grad(u_ex))

    # FE space
    V_el = basix.ufl.element("Lagrange", unf_mesh.basix_cell(), degree)
    V = dolfinx.fem.functionspace(unf_mesh, V_el)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Measures
    dx = ufl.dx(domain=unf_mesh)
    ds_unf = dsu(domain=unf_mesh)
    n_unf = UnfittedNormal(unf_mesh)

    gradu_n = ufl.dot(ufl.grad(u), n_unf)
    gradv_n = ufl.dot(ufl.grad(v), n_unf)

    # Bilinear form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a -= gradu_n * v * ds_unf
    a -= u * gradv_n * ds_unf
    a += (beta / h) * u * v * ds_unf

    # Linear functional
    L = f * v * dx
    L += (beta / h) * u_ex * v * ds_unf
    L -= u_ex * gradv_n * ds_unf

    # Assemble
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()

    b = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Ghost penatly
    if tau != 0.0:
        cut_cells = unf_mesh.get_cut_cells()
        ds_ghost, n_int = ghost_facet_measure(unf_mesh, cut_cells)
        a_ghost = (tau * h) * ufl.jump(ufl.grad(u), n_int) * ufl.jump(ufl.grad(v), n_int) * ds_ghost

        G = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form.__wrapped__(a_ghost))
        G.assemble()

        A.axpy(1.0, G, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
        G.destroy()

    return SystemData(
        A=A,
        b=b,
        V=V,
        unf_mesh=unf_mesh,
        h=h,
        u_ex=u_ex,
        dx=dx,
    )


def solve_system(sys: SystemData) -> SolveData:
    """
    Solve the assemble Poisson system and compute error norms against
    the exact solution.
    
    Returns
    ------------
    SolveData with uh, L2_u, H1_u
    """
    wh = sys.A.createVecRight()
    ksp = PETSc.KSP().create(sys.unf_mesh.comm)
    ksp.setOperators(sys.A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    ksp.solve(sys.b, wh)
 
    uh = dolfinx.fem.Function(sys.V)
    uh.x.array[:] = wh.array_r[:]
    uh.x.scatter_forward()
 
    dx = sys.dx

    def _scalar(form):
        f = qugar.dolfinx.form_custom(form)
        return float(dolfinx.fem.assemble_scalar(
            f, coeffs=f.pack_coefficients()))

    err_u = uh - sys.u_ex
    L2_u = np.sqrt(_scalar(err_u * err_u * dx))
    H1_u = np.sqrt(_scalar(ufl.inner(ufl.grad(err_u), ufl.grad(err_u)) * dx))

    return SolveData(
        uh=uh,
        L2_u=L2_u,
        H1_u=H1_u,
    )


def compute_condition(sys: SystemData) -> float:
    """
    Compute the condition number of the stiffness matrix.

    Parameters
    ------------
    sys:    SystemData from build_system

    Returns
    ------------
    float condition number
    """
    A_ = sys.A.copy()
    A_dense = A_.convert(PETSc.Mat.Type.DENSE).getDenseArray().copy()
    A_.destroy()

    return float(np.linalg.cond(A_dense))


def compute_eigenvalues(sys: SystemData) -> np.ndarray:
    """
    Compute the eigenvalues of the symmetric stiffness matrix.

    Parameters
    ------------
    sys:    SystemData from build_system

    Returns
    ------------
    np.ndarray of eigenvalues sorted in ascending order
    """
    A_ = sys.A.copy()
    A_dense = A_.convert(PETSc.Mat.Type.DENSE).getDenseArray().copy()
    A_.destroy()

    return np.linalg.eigvalsh(A_dense)


def save_vtk(
        sys:        SystemData,
        sol:        SolveData,
        path:       str,
        rep_degree: int=3,
) -> None:
    """
    Reparametrize the solution onto a fitted mesh and save to a VTK file.
 
    Writes the following fields:
        uh      - discrete solution
        u       - exact solution
        err_u   - error (uh - u)
 
    Parameters
    ------------
    sys:        SystemData from build_system
    sol:        SolveData from solve_system
    path:       output file path
    rep_degree: polynomial degree of the reparametrized mesh
    """
    import dolfinx.io
    import qugar.reparam
 
    # Reparametrized mesh
    reparam = qugar.reparam.create_reparam_mesh(
        sys.unf_mesh, degree=rep_degree, levelset=False)
    rep_mesh = reparam.create_mesh()
 
    Q_rep_el = basix.ufl.element("Lagrange", rep_mesh.basix_cell(), rep_degree)
    V_rep = dolfinx.fem.functionspace(rep_mesh, Q_rep_el)
    interp = qugar.reparam.create_interpolation_data(V_rep, sol.uh.function_space)
 
    uh_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    uh_rep.interpolate_nonmatching(sol.uh, *interp)
    uh_rep.name = "uh"
 
    u_ex_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    u_ex_rep.interpolate(
        lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u_ex_rep.name = "u"
 
    err_u_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    err_u_rep.x.array[:] = uh_rep.x.array - u_ex_rep.x.array
    err_u_rep.x.scatter_forward()
    err_u_rep.name = "err_u"
 
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
 
    with dolfinx.io.VTKFile(rep_mesh.comm, str(out), "w") as vtk:
        vtk.write_function([uh_rep, u_ex_rep, err_u_rep])
 
    print(f"Results saved to: {out}")