"""
stokes_problem.py
Core module for the Stokes CutFEM solver.
"""

from dataclasses import dataclass, field
from typing import Optional

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

@dataclass
class SystemData:
    # PETSc objects
    A:          object  # Monolithic matric before pressure pinning
    b:          object  # RHS vector before pressure pinning
    A_uu:       object
    # FE spaces and mesh
    V:          object  # Velocity FE space
    Q:          object  # Pressure FE space
    unf_mesh:   object
    # Pressure pin
    pin_dof:    int
    pin_value:  float
    # Sizes
    n_dofs_V:   int     # Local velocity degrees of freedom
    n_dofs_Q:   int     # Local pressure degrees of freedom
    # Mesh parameter
    h:          float
    # Exact solution
    u_ex:       object
    p_ex:       object
    # Integration measure over physical domain
    dx:         object

@dataclass
class SolveData:
    uh:     object  # Discrete velocity
    ph:     object  # Discrete pressure
    L2_u:   float   # L2 velocity error
    H1_u:   float   # H1 velocity seminorm error
    L2_p:   float   # L2 pressure error
    div_u:  float   # ||div(uh)||_L2

def exact_solution(x):
    """
    Returns (u_ex, p_ex) as UFL expressions on the given spatial coordinates.
    u_x = 20 * x * y^3
    u_y = 5 * x^4 - 5 * y^4
    p = 60 * x^2 * y - 20 * y^3 
    """
    u_ex = ufl.as_vector([20 * x[0] * x[1]**3,
                          5 * x[0]**4 - 5 * x[1]**4])
    p_ex = 60 * x[0]**2 * x[1] - 20 * x[1]**3
    return u_ex, p_ex

def _cut_facet_measure(unf_mesh, cut_cells):
    """
    Build a dS measure restricted to interior facets of cut elements.
    These are the facets over which the ghost penalty acts.
    Returns (dS_cut, n_facet)
    """
    cut_cell_set = set(cut_cells.tolist())
    tdim = unf_mesh.topology.dim
    fdim = tdim - 1
    unf_mesh.topology.create_connectivity(fdim, tdim)
    f2c = unf_mesh.topology.connectivity(fdim, tdim)

    cut_facet_ids = np.array(
        [f for f in range(f2c.num_nodes)
         if len(f2c.links(f)) == 2
         and cut_cell_set.intersection(f2c.links(f).tolist())],
         dtype=np.int32,
    )
    cut_facet_tags = dolfinx.mesh.meshtags(
        unf_mesh, fdim, cut_facet_ids,
        np.ones(len(cut_facet_ids), dtype=np.int32),
    )
    dS_cut = ufl.Measure("dS", domain=unf_mesh,
                         subdomain_data=cut_facet_tags, subdomain_id=1)
    n_facet = ufl.FacetNormal(unf_mesh)
    return dS_cut, n_facet

def build_system(
        n_cells:    int,
        center:     np.ndarray,
        gamma_mu:   float,
        gamma_g:    float,
        gamma_p:    float,
        radius:     float=0.5,
        xmin:       np.ndarray=None,
        xmax:       np.ndarray=None,
        mu:         float=1.0,
        v_degree:   int=2,
        p_degree:   int=1,
) -> SystemData:
    """
    Build and assemble the full Stokes CutFEM system.
    
    Parameters
    ------------
    n_cells:    number of cells per side on the background mesh
    center:     (2,) array, center of immersed disk
    gamma_mu:   Nitsche penalty parameter
    gamma_g:    ghost penalty parameter for velocity and pressure
    gamma_p:    global pressure stabilization parameter
    radius:     disk radius
    xmin:       x-extent of background domain
    xmax:       y-extent of background domain
    mu:         dynamic viscosity
    v_degree:   polynomial degree of velocity space
    p_degree:   polynomial degree of pressure space

    Returns
    ------------
    SystemData  all assembled objects neeed to solve the system
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

    # Exact solution and source
    x = ufl.SpatialCoordinate(unf_mesh)
    u_ex, p_ex = exact_solution(x)
    g = u_ex
    f = dolfinx.fem.Constant(unf_mesh, np.zeros(2, dtype=dtype))

    # FE spaces
    V_el = basix.ufl.element(
        "Lagrange", unf_mesh.basix_cell(), v_degree, shape=(2,))
    Q_el = basix.ufl.element(
        "Lagrange", unf_mesh.basix_cell(), p_degree)
    V = dolfinx.fem.functionspace(unf_mesh, V_el)
    Q = dolfinx.fem.functionspace(unf_mesh, Q_el)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    # Measures

    # Integration over physical domain
    dx = ufl.dx(domain=unf_mesh)
    # Integration over the immersed boundary
    ds_unf = dsu(domain=unf_mesh)
    # Integration over all interior facets
    dS = ufl.Measure("dS", domain=unf_mesh)
    n_unf = UnfittedNormal(unf_mesh)

    cut_cells = unf_mesh.get_cut_cells()
    # Integration over interior facets in F_G
    dS_cut, n_facet = _cut_facet_measure(unf_mesh, cut_cells)

    # Bilinear forms

    A_uu_form = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    A_uu_form -= mu * ufl.inner(ufl.dot(ufl.grad(u), n_unf), v) * ds_unf
    A_uu_form -= mu * ufl.inner(u, ufl.dot(ufl.grad(v), n_unf)) * ds_unf
    A_uu_form += (gamma_mu * mu / h) * ufl.inner(u, v) * ds_unf
    A_uu_form += (gamma_g * mu * h) * ufl.inner(
        ufl.jump(ufl.grad(u), n_facet),
        ufl.jump(ufl.grad(v), n_facet)) * dS_cut


    # j = 2 in stabilisation form
    G2u = ufl.grad(ufl.grad(u))
    G2v = ufl.grad(ufl.grad(v))
    nF  = n_facet("+")

    def d2n_jump(G):
        return (ufl.dot(ufl.dot(G("+"), nF), nF)
                - ufl.dot(ufl.dot(G("-"), nF), nF))

    A_uu_form += (gamma_g * mu * h**3) * ufl.inner(
        d2n_jump(G2u), d2n_jump(G2v)) * dS_cut

    A_up_form = - p * ufl.div(v) * dx
    A_up_form += p * ufl.dot(v, n_unf) * ds_unf

    A_pu_form = ufl.div(u) * q * dx

    # To match the format from Burman
    A_pu_form -= q * ufl.dot(u, n_unf) * ds_unf

    A_pp_form = dolfinx.fem.Constant(unf_mesh, dtype(0.0)) * p * q * dx
    A_pp_form += (gamma_g * h**3 / mu) * ufl.dot(
        ufl.jump(ufl.grad(p), n_facet),
        ufl.jump(ufl.grad(q), n_facet)) * dS_cut
    A_pp_form += (gamma_p / mu) * h**3 * ufl.dot(
        ufl.jump(ufl.grad(p), n_facet),
        ufl.jump(ufl.grad(q), n_facet)) * dS

    # Assemble blocks
    A_uu = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(A_uu_form)); A_uu.assemble()
    A_up = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(A_up_form)); A_up.assemble()
    A_pu = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(A_pu_form)); A_pu.assemble()
    A_pp = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(A_pp_form)); A_pp.assemble()
    
    # Full monolithic matrix
    A_nest = PETSc.Mat().createNest([[A_uu, A_up], [A_pu, A_pp]])
    A_nest.assemble()
    A = PETSc.Mat()
    A_nest.convert("aij", A)
    A.assemble()

    # Right-hand side
    L_u_form = ufl.inner(f, v) * dx
    L_u_form -= mu * ufl.inner(g, ufl.dot(ufl.grad(v), n_unf)) * ds_unf
    L_u_form += (gamma_mu * mu / h) * ufl.inner(g, v) * ds_unf
    L_p_form = dolfinx.fem.Constant(unf_mesh, dtype(0.0)) * q * dx

    # To balance Burman term
    L_p_form -= q * ufl.dot(g, n_unf) * ds_unf

    b_u = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_u_form))
    b_u.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_p = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(L_p_form))
    b_p.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    n_dofs_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    n_dofs_Q = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs

    b = A.createVecRight()
    b.array[:n_dofs_V] = b_u.array_r[:n_dofs_V]
    b.array[n_dofs_V:n_dofs_V + n_dofs_Q] = b_p.array_r[:n_dofs_Q]
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Pressure pin
    full_cells = unf_mesh.get_full_cells()
    pin_dof = Q.dofmap.cell_dofs(full_cells[0])[0]
    pin_coords = Q.tabulate_dof_coordinates()[pin_dof]
    pin_value = float(
        60 * pin_coords[0]**2 * pin_coords[1] - 20 * pin_coords[1]**3)

    return SystemData(
        A=A,
        b=b,
        A_uu=A_uu,
        V=V,
        Q=Q,
        unf_mesh=unf_mesh,
        pin_dof=pin_dof,
        pin_value=pin_value,
        n_dofs_V=n_dofs_V,
        n_dofs_Q=n_dofs_Q,
        h=h,
        u_ex=u_ex,
        p_ex=p_ex,
        dx=dx,
    )

def solve_system(sys: SystemData) -> SolveData:
    """
    Apply the pressure pin, solve the monolithic system with MUMPS,
    and compute error norm against the exact solution.

    Returns
    ------------
    SolveData with uh, ph, L2_u, H1_u, L2_p, div_u
    """
    # Apply pressure pin
    sys.A.zeroRows([sys.n_dofs_V + sys.pin_dof], diag=1.0)
    sys.A.assemble()
    sys.b.setValue(
        sys.n_dofs_V + sys.pin_dof, sys.pin_value,
        addv=PETSc.InsertMode.INSERT)
    sys.b.assemble()

    # Solve
    wh = sys.A.createVecRight()
    ksp = PETSc.KSP().create(sys.unf_mesh.comm)
    ksp.setOperators(sys.A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    ksp.solve(sys.b, wh)

    # Extract solution functions
    uh = dolfinx.fem.Function(sys.V)
    ph = dolfinx.fem.Function(sys.Q)
    uh.x.array[:] = wh.array_r[:sys.n_dofs_V]
    ph.x.array[:] = wh.array_r[sys.n_dofs_V:sys.n_dofs_V + sys.n_dofs_Q]
    uh.x.scatter_forward()
    ph.x.scatter_forward()

    dx = sys.dx

    def _scalar(form):
        f = qugar.dolfinx.form_custom(form)
        return float(dolfinx.fem.assemble_scalar(
            f, coeffs=f.pack_coefficients()))

    vol = _scalar(dolfinx.fem.Constant(sys.unf_mesh, dtype(1.0)) * dx)
    p_mean_h = _scalar(ph * dx) / vol
    p_mean_ex = _scalar(sys.p_ex * dx) / vol

    # Error forms
    err_u = uh - sys.u_ex
    err_p = (ph - p_mean_h) - (sys.p_ex - p_mean_ex)

    L2_u = np.sqrt(_scalar(ufl.inner(err_u, err_u) * dx))
    H1_u = np.sqrt(_scalar(ufl.inner(ufl.grad(err_u), ufl.grad(err_u)) * dx))
    L2_p = np.sqrt(_scalar(err_p * err_p * dx))

    div_u = np.sqrt(_scalar(ufl.div(uh) * ufl.div(uh) * dx))

    return SolveData(
        uh=uh,
        ph=ph,
        L2_u=L2_u,
        H1_u=H1_u,
        L2_p=L2_p,
        div_u=div_u,
    )

def compute_condition(sys: SystemData, max_dofs: int=50_000) -> float:
    """
    Compute the SVD condition number of the full monolithic matrix before
    pinning.

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
    return float(np.abs(sigma[0]) / np.abs(sigma[-1]))

def save_vtk(
        sys:        SystemData,
        sol:        SolveData,
        path:       str,
        rep_degree: int=3,
) -> None:
    """
    Reparametrize the solution onto a fitted mesh and save to a VTK file.

    Writes the following fields:
        uh      - discrete velocity
        u       - exact velocity
        err_u   - velocity error (uh - u)
        uh_mag  - velocity magnitude
        ph      - discrete pressure
        p       - exact pressure
        err_p   - pressure error (ph - p)

    Parameters
    ------------
    sys:        SystemData from build_system
    sol:        SolveData from solve_system
    path:       output file path
    rep_degree: polynomial degree of parametrized mesh
    """
    from pathlib import Path
    import dolfinx.io
    import qugar.reparam

    # Reparametrized mesh
    reparam = qugar.reparam.create_reparam_mesh(
        sys.unf_mesh, degree=rep_degree, levelset=False)
    rep_mesh = reparam.create_mesh()
 
    # Velocity
    V_rep_el = basix.ufl.element(
        "Lagrange", rep_mesh.basix_cell(), rep_degree, shape=(2,))
    V_rep = dolfinx.fem.functionspace(rep_mesh, V_rep_el)
    interp_u = qugar.reparam.create_interpolation_data(V_rep, sol.uh.function_space)
 
    uh_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    uh_rep.interpolate_nonmatching(sol.uh, *interp_u)
    uh_rep.name = "uh"
 
    u_ex_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    u_ex_rep.interpolate(lambda x: np.array([
        20 * x[0] * x[1]**3,
         5 * x[0]**4 - 5 * x[1]**4], dtype=dtype))
    u_ex_rep.name = "u"
 
    err_u_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    err_u_rep.x.array[:] = uh_rep.x.array - u_ex_rep.x.array
    err_u_rep.x.scatter_forward()
    err_u_rep.name = "err_u"
 
    # Velocity magnitude
    S_rep_el = basix.ufl.element(
        "Lagrange", rep_mesh.basix_cell(), rep_degree)
    S_rep    = dolfinx.fem.functionspace(rep_mesh, S_rep_el)
 
    uh_mag_rep = dolfinx.fem.Function(S_rep, dtype=dtype)
    uh_mag_rep.interpolate(
        dolfinx.fem.Expression(
            ufl.sqrt(ufl.inner(uh_rep, uh_rep)),
            S_rep.element.interpolation_points,
        )
    )
    uh_mag_rep.name = "uh_mag"
 
    # Pressure
    Q_rep_el = basix.ufl.element(
        "Lagrange", rep_mesh.basix_cell(), rep_degree)
    Q_rep    = dolfinx.fem.functionspace(rep_mesh, Q_rep_el)
    interp_p = qugar.reparam.create_interpolation_data(Q_rep, sol.ph.function_space)
 
    ph_rep = dolfinx.fem.Function(Q_rep, dtype=dtype)
    ph_rep.interpolate_nonmatching(sol.ph, *interp_p)
    ph_rep.name = "ph"
 
    p_ex_rep = dolfinx.fem.Function(Q_rep, dtype=dtype)
    p_ex_rep.interpolate(
        lambda x: 60 * x[0]**2 * x[1] - 20 * x[1]**3)
    p_ex_rep.name = "p"
 
    err_p_rep = dolfinx.fem.Function(Q_rep, dtype=dtype)
    err_p_rep.x.array[:] = ph_rep.x.array - p_ex_rep.x.array
    err_p_rep.x.scatter_forward()
    err_p_rep.name = "err_p"
 
    # Save
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
 
    with dolfinx.io.VTKFile(rep_mesh.comm, str(out), "w") as vtk:
        vtk.write_function([
            uh_rep, u_ex_rep, err_u_rep, uh_mag_rep,
            ph_rep, p_ex_rep, err_p_rep,
        ])
 
    print(f"Results saved to: {out}")