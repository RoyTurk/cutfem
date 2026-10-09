"""Stationary Navier-Stokes flow around a cylinder (DFG benchmark 2D-1, Re = 20).

    -nu Delta u + (u . grad) u + grad p = 0,   div u = 0   in Omega,

with a parabolic inflow and no-slip walls imposed strongly, no-slip on the
cylinder by Nitsche (viscous and normal penalty), ghost penalty on the cut
facets, solved by Picard iteration. Tunable parameters are fem.Constants,
so they change without recompiling forms (``update_parameters``).
"""

from dataclasses import dataclass, replace

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import ufl
from petsc4py import PETSc

from cutfem import core

# DFG Flow Around Cylinder 2D-1 reference values (Schaefer & Turek, 1996)
DFG_2D1_REF = {"c_D": 5.57953523384, "c_L": 0.010618948146}
QUANTITIES = tuple(DFG_2D1_REF)
ERRORS = tuple(f"err_{q}" for q in QUANTITIES)
TUNABLE = ("nu", "U", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p")


@dataclass(frozen=True)
class Params:
    """Parameters of one Navier-Stokes run."""

    gamma_u: float = 0.1        # velocity ghost penalty
    gamma_p: float = 0.1        # pressure ghost penalty
    gamma_N1: float = 35.0      # Nitsche viscous penalty
    gamma_N2: float = 35.0      # Nitsche normal penalty
    nu: float = 1e-3            # kinematic viscosity
    U_m: float = 0.3            # maximum inflow velocity
    U: float = 0.2              # mean velocity (ghost penalty weight)
    n_y: int = 40               # cells across the channel (see channel_mesh)
    v_degree: int = 2
    p_degree: int = 1
    tol: float = 1e-10          # Picard tolerance on the residual norm
    max_iter: int = 50
    center: tuple = (0.2, 0.2)
    radius: float = 0.05
    xmin: tuple = (0.0, 0.0)
    xmax: tuple = (2.2, 0.41)


@dataclass
class System(core.System):
    """Navier-Stokes system: compiled forms, matrices and Picard state."""

    params: Params
    V: object
    Q: object
    measures: core.CutMeasures
    consts: dict        # fem.Constants: nu, gamma_*, phi
    u_k: object         # advecting velocity of the current Picard step
    uh: object
    ph: object
    a_forms: list       # [[a_uu(u_k), a_up], [a_pu, None]]
    g_forms: list       # [[G_uu, None], [None, G_pp]] (ghost penalty)
    L_forms: list       # [L_u, L_p]
    bcs: list           # strong Dirichlet BCs (inflow and walls)
    G: PETSc.Mat        # ghost penalty matrix


def _phi(nu, U, h):
    """Ghost penalty weight nu + U^2 h^2 / (nu + U h)."""
    return nu + U ** 2 * h ** 2 / (nu + U * h)


def _dirichlet_bcs(mesh, V, p: Params):
    """Parabolic inflow u_x = 4 U_m y (H - y) / H^2 and no-slip walls."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim,
        lambda x: (np.isclose(x[0], p.xmin[0]) | np.isclose(x[1], p.xmin[1])
                   | np.isclose(x[1], p.xmax[1])))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)

    height = p.xmax[1] - p.xmin[1]

    def inflow(x):
        y = x[1] - p.xmin[1]
        return np.vstack((4.0 * p.U_m * y * (height - y) / height ** 2,
                          np.zeros_like(x[0])))

    g = dolfinx.fem.Function(V, name="g_D")
    g.interpolate(inflow)
    return [dolfinx.fem.dirichletbc(g, dofs)]


def channel_mesh(p: Params):
    """Uniform Cartesian mesh of the channel with n_y cells across its height.

    h = H / n_y and n_x = round(L / h), so the cells are square up to the
    rounding of n_x (well below 1 %). Since H = 0.41, the cylinder is never
    aligned with the grid lines for the usual n_y (no tangential cuts).
    Returns ``(mesh, h)`` with h the largest cell side.
    """
    length, height = (b - a for a, b in zip(p.xmin, p.xmax, strict=True))
    n_x = round(length * p.n_y / height)
    return core.disk_mesh([n_x, p.n_y], p.xmin, p.xmax, p.center, p.radius,
                          inside=False)


def build(p: Params) -> System:
    """Build forms, matrices and Picard state for the DFG 2D-1 problem."""
    mesh, h = channel_mesh(p)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.v_degree, shape=(2,))
    Q = core.lagrange_space(mesh, p.p_degree)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    pr, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    def constant(value):
        return dolfinx.fem.Constant(mesh, core.DTYPE(value))

    consts = {key: constant(getattr(p, key))
              for key in ("nu", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p")}
    consts["phi"] = constant(_phi(p.nu, p.U, h))
    nu, gN1, gN2 = consts["nu"], consts["gamma_N1"], consts["gamma_N2"]
    n, dx, ds = m.n, m.dx, m.ds
    zero_vec = dolfinx.fem.Constant(mesh, np.zeros(2, dtype=core.DTYPE))
    g_cyl, f = zero_vec, zero_vec
    u_k = dolfinx.fem.Function(V, name="u_k")

    # Oseen operator at u_k, Nitsche on the cylinder (n points out of the fluid)
    a_uu = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
            + ufl.inner(ufl.dot(ufl.grad(u), u_k), v) * dx
            - nu * ufl.inner(ufl.dot(ufl.grad(u), n), v) * ds
            - nu * ufl.inner(ufl.dot(ufl.grad(v), n), u) * ds
            + (gN1 * nu / h) * ufl.inner(u, v) * ds
            + (gN2 / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds)
    a_up = -pr * ufl.div(v) * dx + pr * ufl.dot(v, n) * ds
    a_pu = ufl.div(u) * q * dx - q * ufl.dot(u, n) * ds
    a_forms = [[core.compile_form(a_uu), core.compile_form(a_up)],
               [core.compile_form(a_pu), None]]

    G_uu = consts["gamma_u"] * consts["phi"] * core.ghost_penalty(u, v, m, h, order=2)
    G_pp = (consts["gamma_p"] / consts["phi"] * h ** 2
            * core.ghost_penalty(pr, q, m, h, order=1))
    g_forms = [[core.compile_form(G_uu, cut=False), None],
               [None, core.compile_form(G_pp, cut=False)]]

    L_u = (ufl.inner(f, v) * dx
           - nu * ufl.inner(g_cyl, ufl.dot(ufl.grad(v), n)) * ds
           + (gN1 * nu / h) * ufl.inner(g_cyl, v) * ds
           + (gN2 / h) * ufl.dot(g_cyl, n) * ufl.dot(v, n) * ds)
    L_p = constant(0.0) * q * dx - q * ufl.dot(g_cyl, n) * ds
    L_forms = [core.compile_form(L_u), core.compile_form(L_p)]

    bcs = _dirichlet_bcs(mesh, V, p)

    # Allocate A with the union of both sparsity patterns, then clear it
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

    return System(mesh=mesh, h=h, A=A, b=b, params=p, V=V, Q=Q, measures=m,
                  consts=consts, u_k=u_k,
                  uh=dolfinx.fem.Function(V, name="uh"),
                  ph=dolfinx.fem.Function(Q, name="ph"),
                  a_forms=a_forms, g_forms=g_forms, L_forms=L_forms,
                  bcs=bcs, G=G)


def _assemble_ghost(s: System):
    """Reassemble the ghost penalty matrix in place (0 on BC rows)."""
    s.G.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(s.G, s.g_forms, bcs=s.bcs, diag=0.0)
    s.G.assemble()


def _assemble_lhs(s: System):
    """Reassemble A(u_k) + G in place."""
    s.A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(s.A, s.a_forms, bcs=s.bcs)
    s.A.assemble()
    s.A.axpy(1.0, s.G, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)


def _assemble_rhs(s: System):
    """Reassemble b in place, with lifting of the strong BCs."""
    with s.b.localForm() as local:
        local.set(0.0)
    dolfinx.fem.petsc.assemble_vector(s.b, s.L_forms)
    bcs1 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(s.a_forms, 1), s.bcs)
    dolfinx.fem.petsc.apply_lifting(s.b, s.a_forms, bcs=bcs1)
    dolfinx.fem.petsc.apply_lifting(s.b, s.g_forms, bcs=bcs1)
    s.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(s.L_forms), s.bcs)
    dolfinx.fem.petsc.set_bc(s.b, bcs0)


def update_parameters(s: System, reset=True, **changes):
    """Change tunable parameters (``TUNABLE``) without recompiling forms.

    The ghost penalty matrix is reassembled; with ``reset`` the iterate is
    set back to zero, so the next solve starts from Stokes.
    """
    unknown = set(changes) - set(TUNABLE)
    if unknown:
        raise KeyError(f"not tunable: {sorted(unknown)}")
    s.params = replace(s.params, **changes)
    for key in ("nu", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p"):
        s.consts[key].value = getattr(s.params, key)
    s.consts["phi"].value = _phi(s.params.nu, s.params.U, s.h)
    _assemble_ghost(s)
    if reset:
        for f in (s.u_k, s.uh, s.ph):
            f.x.array[:] = 0.0


def solve(s: System, max_iter=None, verbose=False):
    """Picard iteration A(u_k) w_{k+1} = b(u_k); returns ``(uh, ph, history)``.

    The first iterate (u_k = 0) is the Stokes solution; ``max_iter=0`` stops
    there. Convergence is checked on the nonlinear residual
    ``r_k = A(u_k) w_k - b(u_k)``; ``history`` holds its norms.
    """
    p = s.params
    max_iter = p.max_iter if max_iter is None else max_iter
    A, b = s.A, s.b
    w = A.createVecRight()
    w.set(0.0)
    r = A.createVecLeft()
    ksp = PETSc.KSP().create(s.mesh.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    history = []
    for k in range(max_iter + 1):
        _assemble_lhs(s)
        _assemble_rhs(s)
        if k > 0:
            A.mult(w, r)
            r.axpy(-1.0, b)
            history.append(r.norm())
            if verbose and s.mesh.comm.rank == 0:
                print(f"Picard {k:3d}: ||r|| = {history[-1]:.3e}", flush=True)
            if history[-1] < p.tol or k == max_iter:
                break

        ksp.setOperators(A)
        ksp.solve(b, w)
        if ksp.getConvergedReason() < 0:
            raise RuntimeError(f"linear solve failed at Picard iteration {k} "
                               f"(KSP reason {ksp.getConvergedReason()})")
        core.split_solution(w, s.uh, s.ph)
        s.u_k.x.array[:] = s.uh.x.array

    if history and history[-1] >= p.tol and s.mesh.comm.rank == 0:
        print(f"Warning: Picard did not converge in {max_iter} iterations "
              f"(||r|| = {history[-1]:.3e})", flush=True)
    for obj in (w, r, ksp):
        obj.destroy()
    return s.uh, s.ph, history


def benchmark(s: System, solution) -> dict:
    """DFG 2D-1 drag and lift coefficients c_D and c_L.

    c_D, c_L = 2 F / (U_bar^2 D) with U_bar = 2/3 U_m and D = 2 r, where F is
    the Nitsche-consistent traction on the cylinder (n out of the fluid):
        F = int_Gamma (p n - nu grad(u) n + gN1 nu / h u + gN2 / h (u.n) n) ds.
    """
    uh, ph, _ = solution
    p, c, h = s.params, s.consts, s.h
    n, ds = s.measures.n, s.measures.ds
    traction = (ph * n - c["nu"] * ufl.dot(ufl.grad(uh), n)
                + (c["gamma_N1"] * c["nu"] / h) * uh
                + (c["gamma_N2"] / h) * ufl.dot(uh, n) * n)
    force = np.array([core.integrate(traction[i] * ds) for i in range(2)])
    scale = 2.0 / ((2.0 / 3.0 * p.U_m) ** 2 * 2.0 * p.radius)
    return {"c_D": scale * force[0], "c_L": scale * force[1]}


def errors(s: System, solution) -> dict:
    """Relative errors of the benchmark quantities against DFG_2D1_REF."""
    values = benchmark(s, solution)
    return {f"err_{q}": abs(values[q] - ref) / abs(ref)
            for q, ref in DFG_2D1_REF.items()}


def inf_sup_matrices(s: System):
    """Matrices of the inf-sup test, as in stokes.inf_sup (mu -> nu).

    Returns ``(N, A_up, G_1, M)``:
        N    velocity norm nu ||grad v||^2_{Omega_T} + nu h^-1 ||v||^2_Gamma,
             identity on the Dirichlet rows (inflow and walls),
        A_up coupling b_h(q, v) = -(q, div v) + (q, v.n)_Gamma, zero
             Dirichlet rows,
        G_1  pressure ghost penalty for gamma_p = 1 (it is linear in gamma_p),
        M    pressure mass matrix on Omega_T, weight 1 / nu.
    The flow solution is not needed: the test only involves the coupling.
    """
    nu, h, m = s.params.nu, s.h, s.measures
    u, v = ufl.TrialFunction(s.V), ufl.TestFunction(s.V)
    pr, q = ufl.TrialFunction(s.Q), ufl.TestFunction(s.Q)

    N = core.assemble_matrix(nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * m.dx,
                             cut=False, bcs=s.bcs)
    core.add_matrix(N, (nu / h) * ufl.inner(u, v) * m.ds, cut=True, bcs=s.bcs)
    A_up = dolfinx.fem.petsc.assemble_matrix(s.a_forms[0][1], bcs=s.bcs)
    A_up.assemble()

    gamma_p = float(s.consts["gamma_p"].value)
    s.consts["gamma_p"].value = 1.0
    G_1 = dolfinx.fem.petsc.assemble_matrix(s.g_forms[1][1])
    G_1.assemble()
    s.consts["gamma_p"].value = gamma_p

    M = core.assemble_matrix((1.0 / nu) * pr * q * m.dx, cut=False)
    return N, A_up, G_1, M


def expected_rates(p: Params) -> dict:
    """No proven orders for the benchmark functionals."""
    return {}


def save_vtk(path, s: System, solution) -> None:
    """Write velocity and pressure for ParaView."""
    uh, ph, _ = solution
    core.save_vtk(path, s.mesh, {"uh": uh, "ph": ph})
