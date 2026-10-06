"""
advdiff_problem.py
Core module for the advection-diffusion CutFEM solver.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from mpi4py import MPI

import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
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
    # FE space and mesh
    V:          object
    unf_mesh:   object
    # Mesh parameter
    h:          float
    # Immersed disk
    center:     np.ndarray
    radius:     float


@dataclass
class SolveData:
    uh:     object  # Discrete solution


def _outer_boundary_dofs(unf_mesh, V, xmin, xmax):
    """
    Locate the DoFs on the outer boundary of the background domain.
    """
    fdim = unf_mesh.topology.dim - 1
    outer_facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, fdim,
        lambda x: (np.isclose(x[0], xmin[0]) | np.isclose(x[0], xmax[0])
                   | np.isclose(x[1], xmin[1]) | np.isclose(x[1], xmax[1]))
    )
    return dolfinx.fem.locate_dofs_topological(V, fdim, outer_facets)


def build_system(
        n_cells:    int,
        mu:         float,
        beta:       np.ndarray,
        gamma:      float,
        gamma_mu:   float,
        gamma_beta: float,
        center:     np.ndarray=None,
        radius:     float=0.1,
        xmin:       np.ndarray=None,
        xmax:       np.ndarray=None,
        g:          float=1.0,
        degree:     int=1,
) -> SystemData:
    """
    Build and assemble the full advection-diffusion CutFEM system.

    Parameters
    ------------
    n_cells:    number of cells per side on the background mesh
    mu:         diffusion coefficient
    beta:       (2,) constant advection field
    gamma:      Nitsche penalty parameter
    gamma_mu:   ghost penalty parameter for cut cells
    gamma_beta: CIP stabilization for convection dominated flows
    center:     (2,) center of immersed disk
    radius:     radius of immersed disk
    xmin:       lower corner of the background domain
    xmax:       upper corner of background domain
    g:          Dirichlet value on the disk boundary
    degree:     polynomial degree of FE space

    Returns
    ------------
    SystemData  all assembled objects needed to solve the system
    """
    if xmin is None:
        xmin = np.array([0.0, 0.0], dtype=dtype)
    if xmax is None:
        xmax = np.array([1.0, 1.0], dtype=dtype)
    if center is None:
        center = np.array([0.2, 0.5], dtype=dtype)
    if beta is None:
        beta = np.array([1.0, 0.0], dtype=dtype)

    center = np.asarray(center, dtype=dtype)
    h = (xmax[0] - xmin[0]) / n_cells

    # Mesh
    impl_func = qugar.impl.create_negative(
        qugar.impl.create_disk(radius, center=center))
    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, impl_func, n_cells, xmin, xmax,
        exclude_empty_cells=True, dtype=dtype
    )

    # Advection vector and disk boundary
    beta_norm = float(np.linalg.norm(beta))
    beta = dolfinx.fem.Constant(unf_mesh, np.asarray(beta, dtype=dtype))
    g = dolfinx.fem.Constant(unf_mesh, dtype(g))

    # FE space
    V_el = basix.ufl.element("Lagrange", unf_mesh.basix_cell(), degree)
    V = dolfinx.fem.functionspace(unf_mesh, V_el)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # Measures

    # Integration over physical domain
    dx = ufl.dx(domain=unf_mesh)
    # Integration over the immersed boundary
    ds_unf = dsu(domain=unf_mesh)
    n_unf = UnfittedNormal(unf_mesh)
    # Integration over all interior facets
    dS = ufl.Measure("dS", domain=unf_mesh)
    n_F = ufl.FacetNormal(unf_mesh)

    # Ghost facets
    cut_cells = unf_mesh.get_cut_cells()
    dS_cut, n_int = ghost_facet_measure(unf_mesh, cut_cells)

    beta_n = ufl.dot(beta, n_unf)
    beta_inflow = 0.5 * (beta_n - abs(beta_n))

    a = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a -= mu * ufl.inner(ufl.grad(u), n_unf) * v * ds_unf
    a -= mu * ufl.inner(ufl.grad(v), n_unf) * u * ds_unf
    a += (gamma * mu) / h * u * v * ds_unf
    a += ufl.dot(beta, ufl.grad(u)) * v * dx
    a -= beta_inflow * u * v * ds_unf

    phi_bar = beta_norm ** 2 * h ** 2 / (mu + beta_norm * h)

    a_stab = (gamma_beta * phi_bar * h) * ufl.jump(ufl.grad(u), n_F) * ufl.jump(ufl.grad(v), n_F) * dS
    a_stab += ((gamma_mu * mu + gamma_beta * phi_bar) * h) * ufl.jump(ufl.grad(u), n_int) * ufl.jump(ufl.grad(v), n_int) * dS_cut

    f = dolfinx.fem.Constant(unf_mesh, dtype(0.0))
    L = f * v * dx
    L -= mu * g * ufl.inner(ufl.grad(v), n_unf) * ds_unf
    L += (gamma * mu) / h * g * v * ds_unf
    L -= beta_inflow * g * v * ds_unf

    outer_dofs = _outer_boundary_dofs(unf_mesh, V, xmin, xmax)
    bcs = [dolfinx.fem.dirichletbc(dtype(0.0), outer_dofs, V)]

    # Assemble
    a_form = dolfinx.fem.form(a)
    stab_form = dolfinx.fem.form.__wrapped__(a_stab)
    L_form = dolfinx.fem.form(L)

    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    G = dolfinx.fem.petsc.assemble_matrix(stab_form, bcs=bcs, diag=0.0)
    G.assemble()
    A.axpy(1.0, G, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    G.destroy()

    b = dolfinx.fem.petsc.assemble_vector(L_form)
    dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
    dolfinx.fem.petsc.apply_lifting(b, [stab_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    return SystemData(
        A=A,
        b=b,
        V=V,
        unf_mesh=unf_mesh,
        h=h,
        center=center,
        radius=radius
    )


def solve_system(sys: SystemData) -> SolveData:
    """
    Solve the assembled advection-diffusion system with MUMPS.

    Returns
    ------------
    SolveData with uh
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

    return SolveData(uh=uh)


def compute_eigenvalues(sys: SystemData) -> np.ndarray:
    """
    Compute the eigenvalues of the symmetric part (A + A^T) / 2 of the
    stiffness matrix. The advection term makes A non-symmetric, and
    x^T A x = x^T (A + A^T)/2 x, so these measure the coercivity of A.

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

    return np.linalg.eigvalsh(0.5 * (A_dense + A_dense.T))


def compute_condition(sys: SystemData, max_dofs: int=50_000) -> float:
    """
    Compute the SVD condition number of the stiffness matrix. A is not
    symmetric, so the ratio of extreme singular values is used.

    Parameters
    ------------
    sys:        SystemData from build_system
    max_dofs:   skips SVD and return np.nan if system is larger than this

    Returns
    ------------
    float - sigma_max / sigma_min, or np.nan, or np.inf
    """
    n, m = sys.A.getSize()
    if n > max_dofs or m > max_dofs:
        return np.nan

    A_ = sys.A.copy()
    A_dense = A_.convert(PETSc.Mat.Type.DENSE).getDenseArray().copy()
    A_.destroy()

    sigma = np.linalg.svd(A_dense, compute_uv=False)

    if sigma[-1] < 1e-300:
        return np.inf
    return float(sigma[0] / sigma[-1])


def sample_line(
        sys:    SystemData,
        sol:    SolveData,
        x:      np.ndarray,
        y:      float,
) -> np.ndarray:
    """
    Evaluate uh at the points (x, y). Points inside the disk (fictitious
    domain) or not found in any cell are returned as np.nan.

    Parameters
    ------------
    sys:    SystemData from build_system
    sol:    SolveData from solve_system
    x:      (n,) x-coordinates of the sampling points
    y:      y-coordinate of the sampling line

    Returns
    ------------
    np.ndarray of uh values at the sampling points
    """
    import dolfinx.geometry

    mesh = sys.unf_mesh
    points = np.zeros((len(x), 3))
    points[:, 0] = x
    points[:, 1] = y

    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points)

    values = np.full(len(x), np.nan)
    inside_disk = np.hypot(x - sys.center[0], y - sys.center[1]) < sys.radius
    idx, cells = [], []
    for i in range(len(x)):
        links = colliding.links(i)
        if len(links) > 0 and not inside_disk[i]:
            idx.append(i)
            cells.append(links[0])

    if idx:
        values[idx] = sol.uh.eval(points[idx], np.array(cells, dtype=np.int32))[:, 0]
    return values


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

    V_rep_el = basix.ufl.element("Lagrange", rep_mesh.basix_cell(), rep_degree)
    V_rep = dolfinx.fem.functionspace(rep_mesh, V_rep_el)
    interp = qugar.reparam.create_interpolation_data(V_rep, sol.uh.function_space)

    uh_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    uh_rep.interpolate_nonmatching(sol.uh, *interp)
    uh_rep.name = "uh"

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with dolfinx.io.VTKFile(rep_mesh.comm, str(out), "w") as vtk:
        vtk.write_function(uh_rep)

    print(f"Results saved to: {out}")