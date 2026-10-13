"""Stationary incompressible Navier-Stokes with CutFEM (problem independent).

    (u . grad) u + grad p - nu Delta u = f,   div u = 0   in Omega,
    u = g_D on Gamma_D^s (strong) and on Gamma (Nitsche),
    nu du/dn - p n = g_N on Gamma_N,

with Gamma_D^s and Gamma_N fitted to the background mesh and Gamma immersed.
Taylor-Hood P2-P1 on the active mesh, Nitsche on Gamma, ghost penalty on the
cut facets; see ``docs/navier_stokes_newton.md`` for the Newton method.

Problem modules (``problems/ns_mms.py``, ``problems/dfg2d1.py``, ...) create
the mesh, the data and the strong boundary facets, and call :func:`build`.
Their ``Params`` subclass :class:`Params`, so every field stays a flat,
sweepable dataclass field.

Strong Dirichlet conditions follow the custom Newton solver of the FEniCSx
tutorial (J. Dokken): the correction satisfies ``du = g_D - u_k`` on the
constrained dofs (``apply_lifting(..., x0=x_k, alpha=-1)`` and
``set_bc(..., x0=x_k, alpha=-1)``), so the initial guess need not satisfy
them. Starting from zero, the first Newton step is the Stokes solve.

Two qugar issues shape the cut forms (``docs/qugar_issues``): no Constant
appears in a cut form that also holds a Function (absent data terms are left
out, not set to zero), and data depending on x enters as a Lagrange Function
(:func:`data_function`), never as a coordinate expression.
"""

from dataclasses import dataclass, field, replace

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.la
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from cutfem import core


@dataclass(frozen=True)
class Params:
    """Discretization and solver parameters shared by all Navier-Stokes problems."""

    nu: float = 1.0             # kinematic viscosity
    U: float = 1.0              # characteristic velocity (ghost penalty weight)
    gamma_N: float = 30.0       # Nitsche viscous penalty gamma_N nu / h
    gamma_n: float = 0.0        # Nitsche normal penalty gamma_n phi_u / h (0: off)
    gamma_u: float = 0.1        # velocity ghost penalty
    gamma_p: float = 0.1        # pressure ghost penalty
    v_degree: int = 2
    p_degree: int = 1
    quadrature_degree: int | None = None  # cut integrals; None: 3 v_degree - 1
    n_picard: int = 0          # Picard steps before switching to Newton
    rtol: float = 1e-10         # Newton: ||R_k|| <= max(atol, rtol ||R_0||)
    atol: float = 1e-12
    max_iter: int = 25
    damping: float = 1.0        # u_{k+1} = u_k + damping du


@dataclass
class System(core.System):
    """Navier-Stokes system; ``A`` is the Jacobian, ``b`` the residual vector.

    After :func:`solve`, ``A`` is the Jacobian at the last iterate (it is
    reassembled at every iteration, unlike the linear problems).
    """

    params: Params
    V: object
    Q: object
    measures: core.CutMeasures  # dx and ds with the fixed quadrature degree
    uh: object                 # current velocity iterate
    ph: object                  # current pressure iterate
    phi_u: float                # ghost penalty weight nu + U^2 h^2 / (nu + U h)
    g: object                   # Nitsche data on Gamma (Function), None if zero
    f: object                   # body force (Function), None if zero
    bcs: list                   # strong Dirichlet BCs (and the pressure pin)
    pinned: bool                # pressure fixed by one dof, zero mean after solve
    forms: dict                 # compiled forms (residual, Jacobians, ghost)
    J: PETSc.Mat                # cut part of the Jacobian (A = J + G)
    G: PETSc.Mat                # ghost penalty matrix, zero on constrained rows/cols
    x: PETSc.Vec                # blocked [u, p] state vector
    history: list = field(default_factory=list)   # residual norms of the last solve


def phi(nu, U, h):
    """Ghost penalty weight phi_u = nu + U^2 h^2 / (nu + U h)."""
    return nu + U ** 2 * h ** 2 / (nu + U * h)


def convection(w, z, v):
    """Integrand of c(w, z, v) = int (w . grad) z . v."""
    return ufl.inner(ufl.dot(ufl.grad(z), w), v)


def strong_bc(V, facets, value):
    """Dirichlet BC on ``facets`` with ``value(x)`` (NumPy callable) interpolated."""
    mesh = V.mesh
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    g = dolfinx.fem.Function(V)
    g.interpolate(value)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
    return dolfinx.fem.dirichletbc(g, dofs)


def slip_bc(V, facets, component):
    """Zero velocity component ``component`` on ``facets`` (free-slip on a face
    normal to that axis); the other components stay free."""
    mesh = V.mesh
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    V_i = V.sub(component)
    dofs = dolfinx.fem.locate_dofs_topological(V_i, fdim, facets)
    return dolfinx.fem.dirichletbc(core.DTYPE(0.0), dofs, V_i)


def boundary_facets(mesh, marker):
    """Exterior facets whose vertices all satisfy ``marker(x)``.

    Use a geometric marker (e.g. ``x[0] == 0``): the facets around the cells
    removed inside an immersed obstacle are exterior facets too.
    """
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    return dolfinx.mesh.locate_entities_boundary(mesh, fdim, marker)


def _pressure_pin(Q):
    """BC fixing one pressure dof of a full cell (all ranks sharing it list it)."""
    comm = Q.mesh.comm
    index_map = Q.dofmap.index_map
    n_owned = index_map.size_local
    dof = None
    for cell in Q.mesh.get_full_cells():
        owned = [d for d in Q.dofmap.cell_dofs(cell) if d < n_owned]
        if owned:
            dof = owned[0]
            break
    candidate = index_map.local_to_global(np.array([dof], dtype=np.int32))[0] \
        if dof is not None else np.iinfo(np.int64).max
    pin = comm.allreduce(int(candidate), op=MPI.MIN)
    if pin == np.iinfo(np.int64).max:
        raise RuntimeError("no full cell to pin the pressure")
    local = index_map.global_to_local(np.array([pin], dtype=np.int64))
    dofs = local[local >= 0].astype(np.int32)
    return dolfinx.fem.dirichletbc(core.DTYPE(0.0), dofs, Q)


def build(params: Params, mesh, h, *, f=None, g=None, strong=(), slip=(),
          neumann=(), neumann_cut=False):
    """Compile the forms and allocate the Newton system.

    Parameters
    ----------
    params : Params
    mesh : qugar unfitted mesh
    h : float
        Mesh size.
    f : UFL vector, optional
        Body force (term omitted if None). Expressions are interpolated into
        P_{k+2} (see :func:`data_function`).
    g : UFL vector, optional
        Dirichlet data on the immersed boundary Gamma (terms omitted if None,
        i.e. no-slip), interpolated like ``f``.
    strong : sequence of (facets, callable)
        Strong Dirichlet conditions on fitted facets, ``value(x)`` NumPy callable.
    slip : sequence of (facets, component)
        Free-slip on fitted facets normal to axis ``component``: that velocity
        component is set to zero strongly, the tangential traction is free.
    neumann : sequence of (ds, g_N)
        Pseudo-traction ``nu du/dn - p n = g_N`` on fitted facets, ``ds`` a
        tagged ``ufl.Measure("ds", ...)``; ``g_N = None`` for do-nothing
        (no term, but the boundary counts as Neumann). Without Neumann
        boundary the pressure is fixed by one dof and shifted to zero mean
        after the solve.
    neumann_cut : bool
        Compile the Neumann terms with cut quadrature (fitted facets cut by
        Gamma, e.g. a gyroid inlet), instead of standard quadrature (facets
        entirely in the fluid, e.g. the DFG outlet). ``g_N`` is then a
        Constant or a Function, never a coordinate expression.
    """
    # One quadrature degree for every cut integral: with the default
    # per-integrand degrees, the residual (degree of c(u, u, v)) and the
    # Jacobian blocks would use different cut rules, i.e. slightly different
    # approximations of Omega, and Newton would only converge linearly.
    degree = params.quadrature_degree or 3 * params.v_degree - 1
    m = core.cut_measures(mesh)
    m = replace(m, dx=m.dx(metadata={"quadrature_degree": degree}),
                ds=m.ds(metadata={"quadrature_degree": degree}))
    dim = mesh.geometry.dim
    V = core.lagrange_space(mesh, params.v_degree, shape=(dim,))
    Q = core.lagrange_space(mesh, params.p_degree)
    uh = dolfinx.fem.Function(V, name="uh")
    ph = dolfinx.fem.Function(Q, name="ph")
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dp, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    zero = dolfinx.fem.Constant(mesh, core.DTYPE(0.0))
    if f is not None:
        f = data_function(f, mesh, params.v_degree + 2, "f")
    if g is not None:
        g = data_function(g, mesh, params.v_degree + 2, "g")
    nu, n, dx, ds = params.nu, m.n, m.dx, m.ds
    phi_u = phi(params.nu, params.U, h)
    w_N = params.gamma_N * nu / h               # viscous Nitsche penalty
    w_n = params.gamma_n * phi_u / h            # normal Nitsche penalty

    def nitsche(u, v, data=None):
        """Gamma terms of a_h, applied to ``u - data`` (``data=None``: to ``u``)."""
        e = u if data is None else u - data
        terms = (-nu * ufl.inner(ufl.dot(ufl.grad(u), n), v)
                 - nu * ufl.inner(ufl.dot(ufl.grad(v), n), e)
                 + w_N * ufl.inner(e, v))
        if params.gamma_n != 0.0:
            terms += w_n * ufl.dot(e, n) * ufl.dot(v, n)
        return terms * ds

    def b_h(v, p):
        return -p * ufl.div(v) * dx + p * ufl.dot(v, n) * ds

    # Residual R(u_h, p_h) on the cut domain (qugar quadrature); data terms
    # only when there is data (no zero Constants next to the Functions)
    F_u = (nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * dx
           + convection(uh, uh, v) * dx
           + nitsche(uh, v, g) + b_h(v, ph))
    F_p = -b_h(uh, q)                           # -b_h(u_h, q) - l_p(q)
    if f is not None:
        F_u -= ufl.inner(f, v) * dx
    if g is not None:
        F_p += q * ufl.dot(g, n) * ds

    # Jacobian at (u_h, p_h); Picard drops c(du, u_h, v)
    J_oseen = (nu * ufl.inner(ufl.grad(du), ufl.grad(v)) * dx
               + convection(uh, du, v) * dx
               + nitsche(du, v))
    J_newton = J_oseen + convection(du, uh, v) * dx
    J_up, J_pu = b_h(v, dp), -b_h(du, q)
    # Empty pressure block: holds the pressure-pin diagonal (dolfinx needs a
    # diagonal block to apply a BC to it)
    J_pp = zero * dp * q * dx

    # Ghost penalty (standard quadrature on the full facets). The weights are
    # Constants: a float 0 would make UFL drop the integrals, and assembling
    # the resulting empty forms segfaults (gamma_u = gamma_p = 0).
    w_u = dolfinx.fem.Constant(mesh, core.DTYPE(params.gamma_u * phi_u))
    w_p = dolfinx.fem.Constant(mesh, core.DTYPE(params.gamma_p / phi_u * h ** 2))
    G_uu = w_u * core.ghost_penalty(du, v, m, h, params.v_degree)
    G_pp = w_p * core.ghost_penalty(dp, q, m, h, params.p_degree)
    R_G = [w_u * core.ghost_penalty(uh, v, m, h, params.v_degree),
           w_p * core.ghost_penalty(ph, q, m, h, params.p_degree)]

    # Neumann data on fitted facets; the zero dx terms only give each block a
    # form (no "everywhere" ds: qugar treats it as a special cut integral)
    R_N = [sum((-ufl.inner(g_N, v) * ds_N for ds_N, g_N in neumann if g_N is not None),
               zero * v[0] * ufl.dx(domain=mesh)),
           zero * q * ufl.dx(domain=mesh)]

    def cut(form):
        return core.compile_form(form)

    def std(form):
        return core.compile_form(form, cut=False)

    neumann_form = cut if neumann_cut else std

    J_up, J_pu, J_pp = cut(J_up), cut(J_pu), cut(J_pp)
    forms = {
        "R": [cut(F_u), cut(F_p)],
        "J_newton": [[cut(J_newton), J_up], [J_pu, J_pp]],
        "J_picard": [[cut(J_oseen), J_up], [J_pu, J_pp]],
        "R_G": [std(R_G[0]), std(R_G[1])],
        "R_N": [neumann_form(R_N[0]), neumann_form(R_N[1])],
        "G": [[std(G_uu), None], [None, std(G_pp)]],
    }

    bcs = [strong_bc(V, facets, value) for facets, value in strong]
    bcs += [slip_bc(V, facets, component) for facets, component in slip]
    pinned = len(neumann) == 0
    if pinned:
        bcs.append(_pressure_pin(Q))

    # The cut forms are assembled into J (their own pattern); A = J + G has
    # the union pattern. Assembling into A directly is not possible: an axpy
    # with a different pattern drops the local-to-global maps of A. J is
    # filled by a first assembly, not by assemble() on the empty matrix,
    # which would squeeze out the preallocated pattern.
    G = dolfinx.fem.petsc.assemble_matrix(forms["G"], bcs=bcs, diag=0.0)
    G.assemble()
    J = dolfinx.fem.petsc.assemble_matrix(forms["J_newton"], bcs=bcs)
    J.assemble()
    A = J.copy()
    A.axpy(1.0, G, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    A.zeroEntries()
    b = dolfinx.fem.petsc.create_vector([V, Q], kind=PETSc.Vec.Type.MPI)
    x = dolfinx.fem.petsc.create_vector([V, Q], kind=PETSc.Vec.Type.MPI)

    return System(mesh=mesh, h=h, A=A, b=b, params=params, V=V, Q=Q, measures=m,
                  uh=uh, ph=ph, phi_u=phi_u, g=g, f=f, bcs=bcs, pinned=pinned,
                  forms=forms, J=J, G=G, x=x)


def data_function(expr, mesh, degree, name):
    """Interpolate a UFL data expression (e.g. of x) into a Lagrange Function.

    Constants and Functions are returned unchanged. qugar cut kernels that
    combine a P1 coefficient (the pressure) with a nonlinear expression of
    the spatial coordinates give wrong values; data given as Functions
    avoid the coordinate expression altogether.
    """
    if isinstance(expr, dolfinx.fem.Function | dolfinx.fem.Constant):
        return expr
    W = core.lagrange_space(mesh, degree, shape=expr.ufl_shape or None)
    w = dolfinx.fem.Function(W, name=name)
    w.interpolate(dolfinx.fem.Expression(expr, W.element.interpolation_points))
    return w


def assemble_residual(s: System):
    """Assemble ``s.b`` = R(u_h, p_h), with constrained entries ``u_h - g_D``.

    Returns the Euclidean norm of the residual.
    """
    dolfinx.fem.petsc.assign([s.uh, s.ph], s.x)
    with s.b.localForm() as local:
        local.set(0.0)
    for key in ("R", "R_G", "R_N"):
        dolfinx.fem.petsc.assemble_vector(s.b, s.forms[key])
    # Constrained columns of J are zeroed: lift du = g_D - u_k into the rhs.
    # The lifting is J (g_D - u_k) on the constrained dofs, zero once the
    # iterate satisfies the BCs (from the first Newton step on): skipped
    # then, as lifting the 3D ghost penalty is expensive.
    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(s.forms["R"]), s.bcs)
    if _bc_mismatch(s, bcs0) > 1e-14 * max(1.0, s.x.norm(PETSc.NormType.INFINITY)):
        J = s.forms["J_newton"]
        bcs1 = dolfinx.fem.bcs_by_block(dolfinx.fem.extract_function_spaces(J, 1),
                                        s.bcs)
        for a in (J, s.forms["G"]):
            dolfinx.fem.petsc.apply_lifting(s.b, a, bcs=bcs1, x0=s.x, alpha=-1.0)
    s.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(s.b, bcs0, s.x, alpha=-1.0)
    return s.b.norm()


def _bc_mismatch(s: System, bcs0):
    """max |g_D - u_k| over the constrained dofs (all ranks)."""
    mismatch = dolfinx.fem.petsc.create_vector([s.V, s.Q], kind=PETSc.Vec.Type.MPI)
    mismatch.set(0.0)
    dolfinx.fem.petsc.set_bc(mismatch, bcs0, s.x, alpha=1.0)
    value = mismatch.norm(PETSc.NormType.INFINITY)
    mismatch.destroy()
    return value


def assemble_jacobian(s: System, picard=False):
    """Assemble ``s.A`` = J(u_h) + G (Picard: without c(du, u_h, v))."""
    s.J.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(
        s.J, s.forms["J_picard" if picard else "J_newton"], bcs=s.bcs)
    s.J.assemble()
    s.A.zeroEntries()
    for M in (s.J, s.G):
        s.A.axpy(1.0, M, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)


def solve(s: System, verbose=False):
    """Newton iteration (optionally Picard first); returns ``(uh, ph)``.

    Starts from the current ``s.uh, s.ph`` (zero after :func:`build`, so the
    first step is a Stokes solve). Stops when
    ``||R_k|| <= max(atol, rtol ||R_0||)``; raises RuntimeError otherwise.
    The residual norms are stored in ``s.history``.
    """
    params = s.params
    ksp = PETSc.KSP().create(s.mesh.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    delta = s.A.createVecRight()
    du = dolfinx.fem.Function(s.V)
    dp = dolfinx.fem.Function(s.Q)

    s.history = []
    try:
        for k in range(params.max_iter + 1):
            norm = assemble_residual(s)
            s.history.append(norm)
            if verbose and s.mesh.comm.rank == 0:
                kind = "Picard" if k < params.n_picard else "Newton"
                print(f"  {kind} {k:2d}: ||R|| = {norm:.3e}", flush=True)
            if norm <= max(params.atol, params.rtol * s.history[0]):
                break
            if k == params.max_iter:
                raise RuntimeError(f"Newton did not converge in {params.max_iter} "
                                   f"iterations (||R|| = {norm:.3e})")

            assemble_jacobian(s, picard=k < params.n_picard)
            s.b.scale(-1.0)
            ksp.setOperators(s.A)
            ksp.solve(s.b, delta)
            if ksp.getConvergedReason() < 0:
                raise RuntimeError(f"linear solve failed at iteration {k} "
                                   f"(KSP reason {ksp.getConvergedReason()})")
            core.split_solution(delta, du, dp)
            s.uh.x.array[:] += params.damping * du.x.array
            s.ph.x.array[:] += params.damping * dp.x.array
    finally:
        for obj in (ksp, delta):
            obj.destroy()

    if s.pinned:
        area = core.integrate(dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0))
                              * s.measures.dx)
        s.ph.x.array[:] -= core.integrate(s.ph * s.measures.dx) / area
        s.ph.x.scatter_forward()
    return s.uh, s.ph


# Post-processing


def traction(s: System, solution):
    """Nitsche flux on Gamma: force density of the fluid on the obstacle.

    With eta = -n pointing into the fluid,
        t = (nu grad u - p I) eta + gamma_N nu / h (u - g)
            + gamma_n phi_u / h ((u - g) . n) n,
    i.e. the benchmark traction plus the Nitsche penalty: the boundary flux
    the discrete equations impose.
    """
    uh, ph = solution
    params, n = s.params, s.measures.n
    e = uh if s.g is None else uh - s.g
    t = (ph * n - params.nu * ufl.dot(ufl.grad(uh), n)
         + (params.gamma_N * params.nu / s.h) * e)
    if params.gamma_n != 0.0:
        t += (params.gamma_n * s.phi_u / s.h) * ufl.dot(e, n) * n
    return t


def force(s: System, solution):
    """Force of the fluid on Gamma, ``int_Gamma traction ds`` (one per direction)."""
    t = traction(s, solution)
    return np.array([core.integrate(t[i] * s.measures.ds)
                     for i in range(s.mesh.geometry.dim)])


def volume_force(s: System, solution, v_e):
    """Variational force -[nu (grad u, grad v_e) + c(u, u, v_e) - (p, div v_e) - (f, v_e)].

    ``v_e`` must be a constant vector e on every cut cell and its facet
    neighbours and vanish on the fitted boundaries (see
    :func:`force_test_function`). Then it equals ``force(s, solution) . e``
    up to the Newton residual: a consistency check of :func:`force`.
    """
    uh, ph = solution
    dx = s.measures.dx
    integrand = (s.params.nu * ufl.inner(ufl.grad(uh), ufl.grad(v_e))
                 + convection(uh, uh, v_e) - ph * ufl.div(v_e))
    if s.f is not None:
        integrand -= ufl.inner(s.f, v_e)
    return -core.integrate(integrand * dx)


def force_test_function(s: System, e):
    """Function of V equal to ``e`` on the cut cells and their facet neighbours, 0 elsewhere."""
    mesh = s.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    c2f = mesh.topology.connectivity(tdim, tdim - 1)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)

    flags = dolfinx.la.vector(mesh.topology.index_map(tdim), dtype=np.float64)
    flags.array[:] = 0.0
    flags.array[mesh.get_cut_cells()] = 1.0
    flags.scatter_forward()
    cut = np.flatnonzero(flags.array > 0.5)
    patch = set(cut.tolist())
    for cell in cut:
        for facet in c2f.links(cell):
            patch.update(f2c.links(facet).tolist())

    v_e = dolfinx.fem.Function(s.V)
    bs = s.V.dofmap.bs
    values = v_e.x.array.reshape(-1, bs)
    for cell in patch:
        values[s.V.dofmap.cell_dofs(cell)] = e
    v_e.x.scatter_forward()
    return v_e


def free_dofs(s: System):
    """Owned global rows of ``s.A`` not constrained by a Dirichlet BC (or pin)."""
    n_u = s.V.dofmap.index_map.size_local * s.V.dofmap.index_map_bs
    n_owned = n_u + s.Q.dofmap.index_map.size_local
    constrained = []
    for i, bc in enumerate(s.bcs):
        dofs, n_owned_bc = bc.dof_indices()
        pin = s.pinned and i == len(s.bcs) - 1       # the pin is the last BC
        constrained.append((n_u if pin else 0) + dofs[:n_owned_bc])
    free = np.setdiff1d(np.arange(n_owned), np.concatenate([[], *constrained]))
    return (s.A.getOwnershipRange()[0] + free).astype(PETSc.IntType)


class _Inverse:
    """Shell matrix M^{-1}: products are solves with one MUMPS LU of M."""

    def __init__(self, ksp):
        self.ksp = ksp

    def mult(self, mat, x, y):
        self.ksp.solve(x, y)

    def multTranspose(self, mat, x, y):
        self.ksp.solveTranspose(x, y)


def _largest_singular_value(M, tol):
    from slepc4py import SLEPc

    svd = SLEPc.SVD().create(M.getComm())
    svd.setOperators(M)
    svd.setType(SLEPc.SVD.Type.TRLANCZOS)
    svd.setImplicitTranspose(True)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setDimensions(nsv=1)
    svd.setTolerances(tol=tol, max_it=1000)
    svd.solve()
    if svd.getConverged() < 1:
        raise RuntimeError("SLEPc SVD did not converge")
    sigma = svd.getValue(0)
    svd.destroy()
    return sigma


def condition_number(s: System, tol=1e-8):
    """2-norm condition number of the Jacobian ``s.A`` on the free unknowns.

    Rows and columns of the strong Dirichlet dofs (and of the pressure pin)
    are removed: their unit diagonal is unrelated to the scale of the
    discrete operator (about nu) and would set sigma_max or sigma_min.
    sigma_max by SLEPc Lanczos bidiagonalization on A_free; sigma_min as
    1 / sigma_max(A_free^{-1}), the inverse applied by one MUMPS LU (the
    smallest singular values are clustered, the largest of the inverse is
    well separated). Call after :func:`solve`, so that ``s.A`` is the
    Jacobian at the solution. Returns ``(kappa, sigma_max, sigma_min)``.
    """
    rows = PETSc.IS().createGeneral(free_dofs(s), comm=s.mesh.comm)
    A = s.A.createSubMatrix(rows, rows)
    ksp = PETSc.KSP().create(s.mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    inverse = PETSc.Mat().createPython(A.getSizes(), _Inverse(ksp), comm=s.mesh.comm)
    inverse.setUp()
    try:
        sigma_max = _largest_singular_value(A, tol)
        sigma_min = 1.0 / _largest_singular_value(inverse, tol)
    finally:
        for obj in (inverse, ksp, A, rows):
            obj.destroy()
    return sigma_max / sigma_min, sigma_max, sigma_min


def cell_reynolds(s: System, uh):
    """Largest cell Reynolds number h max|u_h| / (2 nu) over the velocity dofs."""
    values = uh.x.array.reshape(-1, s.V.dofmap.bs)
    local = float(np.max(np.linalg.norm(values, axis=1), initial=0.0))
    speed = s.mesh.comm.allreduce(local, op=MPI.MAX)
    return s.h * speed / (2.0 * s.params.nu)
