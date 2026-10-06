"""
navierstokes_problem.py
Core module for the stationary Navier-Stokes CutFEM solver.
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


# DFG Flow Around Cylinder 2D-1, laminar case Re = 20
DFG_2D1_REF = {
    "c_D": 5.57953523384,
    "c_L": 0.010618948146,
    "dp":  0.11752016697,
}


@dataclass
class NSParams:
    # Ghost penalty
    gamma_u:    float   # Velocity ghost penalty
    gamma_p:    float   # Pressure ghost penalty
    # Nitsche
    gamma_N1:   float   # Viscous penalty
    gamma_N2:   float   # Normal penalty
    # Physics
    nu:         float=1e-3  # Kinematic viscosity
    U_m:        float=0.3   # Maximum inflow velocity
    U:          float=0.2   # Mean velocity
    # Geometry
    xmin:       tuple=(0.0, 0.0)
    xmax:       tuple=(2.2, 0.41)
    center:     tuple=(0.2, 0.2)
    radius:     float=0.05
    # Discretization
    ref:        float=1.0
    v_degree:   int=2
    p_degree:   int=1
    tol:        float=1e-10
    max_iter:   int=50


@dataclass
class NSSystem:
    # Mesh and spaces
    unf_mesh:   object
    V:          object
    Q:          object
    n_dofs_V:   int         # Owned velocity DoFs
    n_dofs_Q:   int         # Owned pressure DoFs
    h:          float
    params:     NSParams
    consts:     dict        # fem.Constant objects: nu, gamma_*, phi
    # Picard state
    u_k:        object      # Advecting velocity
    uh:         object      # Current velocity iterate
    ph:         object      # Current pressure iterate
    # Compiled blocked forms
    a_forms:    list        # [[a_uu(u_k), a_up], [a_pu, None]]
    g_forms:    list        # [[G_uu, None], [None, G_pp]]  (ghost penalty)
    L_forms:    list        # [L_u, L_p]
    # Boundary conditions
    bcs:        list        # Strong Dirichlet BCs (inflow + walls)
    # Matrices allocated once
    A:          object      # Monolithic matrix
    G:          object      # Ghost-penalty matrix
    b:          object      # monolithic RHS
    # Measures
    dx:         object
    ds_unf:     object
    n_unf:      object

def _phi_u(nu: float, U: float, h: float) -> float:
    """
    Ghost-penalty weight phi_u = nu + U^2 h^2 / (nu + U h).
    Reduces to nu (Stokes) for U -> 0 and to ~ U h for large Peclet.
    """
    return nu + U ** 2 * h ** 2 / (nu + U * h)

def _inflow_profile(prm: NSParams):
    """
    Parabolic inflow u_x = 4 U_m y (H - y) / H^2, u_y = 0.
    """
    H = prm.xmax[1] - prm.xmin[1]

    def g(x):
        y = x[1] - prm.xmin[1]
        return np.vstack((4.0 * prm.U_m * y * (H - y) / H ** 2,
                          np.zeros_like(x[0])))
    return g

def _dirichlet_bcs(unf_mesh, V, prm: NSParams) -> list:
    """
    Strong Dirichlet BC on the inflow and walls.
    """
    tdim = unf_mesh.topology.dim
    fdim = tdim - 1
    unf_mesh.topology.create_connectivity(fdim, tdim)

    facets = dolfinx.mesh.locate_entities_boundary(
        unf_mesh, fdim,
        lambda x: (np.isclose(x[0], prm.xmin[0])
                   | np.isclose(x[1], prm.xmin[1])
                   | np.isclose(x[1], prm.xmax[1])))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)

    g = dolfinx.fem.Function(V, name="g_D")
    g.interpolate(_inflow_profile(prm))
    return [dolfinx.fem.dirichletbc(g, dofs)]

def _assemble_rhs(sys: NSSystem) -> None:
    """
    Assemble the monolithic RHS in place, with lifting for the strong BCs.
    """
    b = sys.b
    with b.localForm() as loc:
        loc.set(0.0)
    dolfinx.fem.petsc.assemble_vector(b, sys.L_forms)

    bcs1 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(sys.a_forms, 1), sys.bcs)
    dolfinx.fem.petsc.apply_lifting(b, sys.a_forms, bcs=bcs1)
    dolfinx.fem.petsc.apply_lifting(b, sys.g_forms, bcs=bcs1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(sys.L_forms), sys.bcs)
    dolfinx.fem.petsc.set_bc(b, bcs0)

def _assemble_lhs(sys: NSSystem) -> None:
    """
    Reassemble A(u_k) in place: blocked qugar forms (with BC rows/columns
    eliminated and 1 on the diagonal) plus the static ghost penalty G.
    """
    A = sys.A
    A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(A, sys.a_forms, bcs=sys.bcs)
    A.assemble()
    A.axpy(1.0, sys.G, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)

def _assemble_ghost(sys: NSSystem) -> None:
    """Reassemble the static ghost-penalty matrix in place (diag 0 on BC rows)."""
    sys.G.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(sys.G, sys.g_forms, bcs=sys.bcs, diag=0.0)
    sys.G.assemble()

def build_system(prm: NSParams) -> NSSystem:
    """
    Build the stationary Navier-Stokes CutFEM system for Picard iteration.
    """
    xmin = np.array(prm.xmin, dtype=dtype)
    xmax = np.array(prm.xmax, dtype=dtype)
    center = np.array(prm.center, dtype=dtype)

    # Mesh size
    n_x = max(1, int(np.floor(220 * prm.ref + 0.5)))
    n_y = max(1, int(np.floor(41 * prm.ref + 0.5)))
    hx = (xmax[0] - xmin[0]) / n_x
    hy = (xmax[1] - xmin[1]) / n_y
    if max(hx, hy) / min(hx, hy) > 1.1:
        raise ValueError(f"Cells too anisotropic: hx = {hx:.4e}, hy = {hy:.4e}")
    h = float(max(hx, hy))

    # Mesh
    impl_func = qugar.impl.create_negative(
        qugar.impl.create_disk(prm.radius, center=center))
    unf_mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, impl_func, [n_x, n_y], xmin, xmax,
        exclude_empty_cells=True, dtype=dtype
    )

    # FE spaces
    V_el = basix.ufl.element(
        "Lagrange", unf_mesh.basix_cell(), prm.v_degree, shape=(2,))
    Q_el = basix.ufl.element(
        "Lagrange", unf_mesh.basix_cell(), prm.p_degree)
    V = dolfinx.fem.functionspace(unf_mesh, V_el)
    Q = dolfinx.fem.functionspace(unf_mesh, Q_el)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    p, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    n_dofs_V = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    n_dofs_Q = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs

    # Measures
    dx = ufl.dx(domain=unf_mesh)            # Fluid domain
    ds_unf = dsu(domain=unf_mesh)           # Cylinder boundary
    n_unf = UnfittedNormal(unf_mesh)        # Normal outward from the fluid
    cut_cells = unf_mesh.get_cut_cells()
    dS_cut, n_facet = ghost_facet_measure(unf_mesh, cut_cells)
    nF = n_facet("+")

    # Tunable parameters as Constants
    def C(val):
        return dolfinx.fem.Constant(unf_mesh, dtype(val))

    consts = {
        "nu":       C(prm.nu),
        "gamma_N1": C(prm.gamma_N1),
        "gamma_N2": C(prm.gamma_N2),
        "gamma_u":  C(prm.gamma_u),
        "gamma_p":  C(prm.gamma_p),
        "phi":      C(_phi_u(prm.nu, prm.U, h)),
    }
    nu = consts["nu"]
    gN1, gN2 = consts["gamma_N1"], consts["gamma_N2"]
    gu, gp, phi = consts["gamma_u"], consts["gamma_p"], consts["phi"]

    # Data
    f = dolfinx.fem.Constant(unf_mesh, np.zeros(2, dtype=dtype))
    g_cyl = dolfinx.fem.Constant(unf_mesh, np.zeros(2, dtype=dtype))
    zero = dolfinx.fem.Constant(unf_mesh, dtype(0.0))

    # Advecting velocity
    u_k = dolfinx.fem.Function(V, name="u_k")

    # Bilinear forms (qugar, physical domain)
    a_uu = nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a_uu += ufl.inner(ufl.dot(ufl.grad(u), u_k), v) * dx
    a_uu -= nu * ufl.inner(ufl.dot(ufl.grad(u), n_unf), v) * ds_unf
    a_uu -= nu * ufl.inner(ufl.dot(ufl.grad(v), n_unf), u) * ds_unf
    a_uu += (gN1 * nu / h) * ufl.inner(u, v) * ds_unf
    a_uu += (gN2 / h) * ufl.dot(u, n_unf) * ufl.dot(v, n_unf) * ds_unf

    a_up = - p * ufl.div(v) * dx
    a_up += p * ufl.dot(v, n_unf) * ds_unf

    a_pu = ufl.div(u) * q * dx
    a_pu -= q * ufl.dot(u, n_unf) * ds_unf

    a_forms = [
        [dolfinx.fem.form(a_uu), dolfinx.fem.form(a_up)],
        [dolfinx.fem.form(a_pu), None],
    ]

    # Ghost penalty
    def jump_dn(w):
        """[[d_n w]]"""
        return (ufl.dot(ufl.grad(w)("+"), nF)
                - ufl.dot(ufl.grad(w)("-"), nF))

    def jump_dn2(w):
        """[[d_n^2 w]]"""
        H = ufl.grad(ufl.grad(w))
        return (ufl.dot(ufl.dot(H("+"), nF), nF)
                - ufl.dot(ufl.dot(H("-"), nF), nF))

    G_uu = gu * phi * (
        h * ufl.inner(jump_dn(u), jump_dn(v))
        + h ** 3 * ufl.inner(jump_dn2(u), jump_dn2(v))) * dS_cut
    G_pp = gp * (h ** 3 / phi) * jump_dn(p) * jump_dn(q) * dS_cut

    g_forms = [
        [dolfinx.fem.form.__wrapped__(G_uu), None],
        [None, dolfinx.fem.form.__wrapped__(G_pp)],
    ]

    # Linear forms
    L_u = ufl.inner(f, v) * dx
    L_u -= nu * ufl.inner(g_cyl, ufl.dot(ufl.grad(v), n_unf)) * ds_unf
    L_u += (gN1 * nu / h) * ufl.inner(g_cyl, v) * ds_unf
    L_u += (gN2 / h) * ufl.dot(g_cyl, n_unf) * ufl.dot(v, n_unf) * ds_unf

    L_p = zero * q * dx
    L_p -= q * ufl.dot(g_cyl, n_unf) * ds_unf

    L_forms = [dolfinx.fem.form(L_u), dolfinx.fem.form(L_p)]

    # Boundary conditions
    bcs = _dirichlet_bcs(unf_mesh, V, prm)

    G = dolfinx.fem.petsc.create_matrix(g_forms)
    dolfinx.fem.petsc.assemble_matrix(G, g_forms, bcs=bcs, diag=0.0)
    G.assemble()

    A = dolfinx.fem.petsc.create_matrix(a_forms)
    dolfinx.fem.petsc.assemble_matrix(A, a_forms, bcs=bcs)
    A.assemble()

    A.axpy(1.0, G, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    A.zeroEntries()

    b = dolfinx.fem.petsc.create_vector(
        dolfinx.fem.extract_function_spaces(L_forms), kind=PETSc.Vec.Type.MPI)

    # Iterates
    uh = dolfinx.fem.Function(V, name="uh")
    ph = dolfinx.fem.Function(Q, name="ph")

    return NSSystem(
        unf_mesh=unf_mesh, V=V, Q=Q,
        n_dofs_V=n_dofs_V, n_dofs_Q=n_dofs_Q, h=h,
        params=prm, consts=consts,
        u_k=u_k, uh=uh, ph=ph,
        a_forms=a_forms, g_forms=g_forms, L_forms=L_forms,
        bcs=bcs, A=A, G=G, b=b,
        dx=dx, ds_unf=ds_unf, n_unf=n_unf,
    )

def update_parameters(sys: NSSystem, reset: bool=True, **kwargs) -> None:
    """
    Change tunable parameters without recompiling any form.
    The ghost-penalty matrix is reassembled; with reset=True the iterate
    is set back to zero so the next Picard solve starts from Stokes.
    """
    prm = sys.params
    allowed = {"nu", "U", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p"}
    for key, val in kwargs.items():
        if key not in allowed:
            raise KeyError(f"Unknown parameter '{key}'")
        setattr(prm, key, val)

    for key in ("nu", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p"):
        sys.consts[key].value = getattr(prm, key)
    sys.consts["phi"].value = _phi_u(prm.nu, prm.U, sys.h)

    _assemble_ghost(sys)

    if reset:
        for fun in (sys.u_k, sys.uh, sys.ph):
            fun.x.array[:] = 0.0

def picard_solve(sys: NSSystem, verbose: bool=True):
    """
    Solve the stationary Navier-Stokes system by Picard iteration.

    Iteration k:
        A(u_k) w_{k+1} = b(u_k),
    with the convective term <(u_k . grad) u, v> in the velocity block;
    b depends on u_k only through the lifting of the strong BCs.
    The first iteration (u_k = 0) gives the Stokes solution.

    Convergence is checked on the full nonlinear residual
        r_k = A(u_k) w_k - b(u_k),
    which is the Navier-Stokes residual at the current iterate
    (w_k satisfies the strong BCs, so the eliminated columns are consistent).

    Returns
    ------------
    (uh, ph, history) with history the list of residual norms
    """
    prm = sys.params
    nV, nQ = sys.n_dofs_V, sys.n_dofs_Q
    comm = sys.unf_mesh.comm
    A, b = sys.A, sys.b

    w = A.createVecRight()
    w.set(0.0)
    r = A.createVecLeft()

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    history = []
    converged = False

    for k in range(prm.max_iter + 1):
        # A(u_k), b(u_k), assembled in place
        _assemble_lhs(sys)
        _assemble_rhs(sys)

        # Nonlinear residual at the current iterate (skip for k = 0: w = 0)
        if k > 0:
            A.mult(w, r)
            r.axpy(-1.0, b)
            res = r.norm()
            history.append(res)
            if verbose and comm.rank == 0:
                print(f"Picard {k:3d}: ||r|| = {res:.3e}")
            if res < prm.tol:
                converged = True
                break
            if k == prm.max_iter:
                break

        # Solve the Oseen system (new numeric factorization of the same pattern)
        ksp.setOperators(A)
        ksp.solve(b, w)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(
                f"Linear solve failed at Picard iteration {k}: "
                f"KSP reason {ksp.getConvergedReason()}, "
                f"PC failed reason {ksp.getPC().getFailedReason()}")

        # Split the blocked vector: [owned u | owned p]
        sys.uh.x.array[:nV] = w.array_r[:nV]
        sys.ph.x.array[:nQ] = w.array_r[nV:nV + nQ]
        sys.uh.x.scatter_forward()
        sys.ph.x.scatter_forward()
        sys.u_k.x.array[:] = sys.uh.x.array
        sys.u_k.x.scatter_forward()

    if not converged and history and comm.rank == 0:
        print(f"Warning: Picard did not converge in {prm.max_iter} iterations "
              f"(last ||r|| = {history[-1]:.3e})")

    for obj in (w, r, ksp):
        obj.destroy()

    return sys.uh, sys.ph, history

def save_vtk(
        sys:        NSSystem,
        path:       str,
        rep_degree: int=3,
) -> None:
    """
    Reparametrize the current solution (sys.uh, sys.ph) onto a fitted mesh
    and save to a VTK file.

    Writes the following fields:
        uh      - discrete velocity
        uh_mag  - velocity magnitude
        ph      - discrete pressure
    """
    import dolfinx.io
    import qugar.reparam

    reparam = qugar.reparam.create_reparam_mesh(
        sys.unf_mesh, degree=rep_degree, levelset=False)
    rep_mesh = reparam.create_mesh()

    # Velocity
    V_rep_el = basix.ufl.element(
        "Lagrange", rep_mesh.basix_cell(), rep_degree, shape=(2,))
    V_rep = dolfinx.fem.functionspace(rep_mesh, V_rep_el)
    interp_u = qugar.reparam.create_interpolation_data(V_rep, sys.uh.function_space)

    uh_rep = dolfinx.fem.Function(V_rep, dtype=dtype)
    uh_rep.interpolate_nonmatching(sys.uh, *interp_u)
    uh_rep.name = "uh"

    # Velocity magnitude
    S_rep_el = basix.ufl.element(
        "Lagrange", rep_mesh.basix_cell(), rep_degree)
    S_rep = dolfinx.fem.functionspace(rep_mesh, S_rep_el)

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
    Q_rep = dolfinx.fem.functionspace(rep_mesh, Q_rep_el)
    interp_p = qugar.reparam.create_interpolation_data(Q_rep, sys.ph.function_space)

    ph_rep = dolfinx.fem.Function(Q_rep, dtype=dtype)
    ph_rep.interpolate_nonmatching(sys.ph, *interp_p)
    ph_rep.name = "ph"

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with dolfinx.io.VTKFile(rep_mesh.comm, str(out), "w") as vtk:
        vtk.write_function([uh_rep, uh_mag_rep, ph_rep])

    if rep_mesh.comm.rank == 0:
        print(f"Results saved to: {out}")


def _scalar(form, comm) -> float:
    """Assemble a scalar unfitted form, summed over all ranks."""
    f = qugar.dolfinx.form_custom(form)
    local = dolfinx.fem.assemble_scalar(f, coeffs=f.pack_coefficients())
    return float(comm.allreduce(local, op=MPI.SUM))
 
 
def _eval_scalar_at(func, points_2d: np.ndarray) -> np.ndarray:
    """
    Evaluate a scalar Function at 2D points. Each point is evaluated on the
    rank owning a cell that contains it; results are combined over ranks.
    """
    import dolfinx.geometry
 
    mesh = func.function_space.mesh
    comm = mesh.comm
    pts = np.zeros((len(points_2d), 3))
    pts[:, :2] = points_2d
 
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, pts)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, pts)
    n_owned = mesh.topology.index_map(mesh.topology.dim).size_local
 
    values = np.zeros(len(pts))
    found = np.zeros(len(pts))
    for i in range(len(pts)):
        cells = [c for c in colliding.links(i) if c < n_owned]
        if cells:
            values[i] = np.ravel(func.eval(pts[i:i + 1], np.array(cells[:1], dtype=np.int32)))[0]            
            found[i] = 1.0
 
    # A point on a cell edge shared by two ranks is found twice: average
    values = comm.allreduce(values, op=MPI.SUM)
    found = comm.allreduce(found, op=MPI.SUM)
    return values / found
 
 
def postprocess(sys: NSSystem, verbose: bool=True) -> dict:
    """
    DFG 2D-1 benchmark quantities:
 
        c_D, c_L = 2 F / (U_bar^2 D),   U_bar = 2/3 U_m,  D = 2 r,
        dp       = p(x_c - r, y_c) - p(x_c + r, y_c).
 
    The force on the cylinder is the Nitsche-consistent boundary traction,
    with n pointing OUT of the fluid (into the cylinder) and g = 0:
 
        F = int_Gamma ( p n - nu grad(u) n
                        + gN1 nu / h  u + gN2 / h (u . n) n ) ds
    """
    prm, c = sys.params, sys.consts
    n, ds, h = sys.n_unf, sys.ds_unf, sys.h
    uh, ph = sys.uh, sys.ph
    comm = sys.unf_mesh.comm
 
    # Forces
    traction = (ph * n - c["nu"] * ufl.dot(ufl.grad(uh), n)
                + (c["gamma_N1"] * c["nu"] / h) * uh
                + (c["gamma_N2"] / h) * ufl.dot(uh, n) * n)
    F = np.array([_scalar(traction[i] * ds, comm) for i in range(2)])
 
    U_bar = 2.0 / 3.0 * prm.U_m
    scale = 2.0 / (U_bar ** 2 * 2.0 * prm.radius)
 
    # Pressure difference
    xc, yc, r = prm.center[0], prm.center[1], prm.radius
    p_front, p_back = _eval_scalar_at(ph, np.array([[xc - r, yc], [xc + r, yc]]))
 
    res = {
        "c_D": scale * F[0],
        "c_L": scale * F[1],
        "dp":  float(p_front - p_back),
    }
 
    if verbose and comm.rank == 0:
        print(f"\nDFG 2D-1 results (h = {h:.3e})")
        print(f"  {'quantity':<10}{'computed':>16}{'reference':>16}{'rel. error':>14}")
        for key, val in res.items():
            ref = DFG_2D1_REF[key]
            print(f"  {key:<10}{val:>16.8f}{ref:>16.8f}{abs(val - ref) / abs(ref):>14.3e}")
 
    return res