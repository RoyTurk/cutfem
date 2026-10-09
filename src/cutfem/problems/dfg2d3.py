"""Unsteady Navier-Stokes flow around a cylinder (DFG benchmark 2D-3, Re <= 100).

    u_t - nu Delta u + (u . grad) u + grad p = 0,   div u = 0   in Omega,

on t in [0, 8] with the inflow amplitude U(t) = U_m sin(pi t / 8), so the
mean inflow velocity peaks at 1 (Re = 100) at t = 4. Same spatial
discretization as :mod:`cutfem.problems.dfg2d1` (Taylor-Hood, Nitsche on the
cylinder, ghost penalty on the cut facets); BDF2 in time (backward Euler for
the first step), and a Newton solve per step with the Jacobian obtained by
``ufl.derivative``.
"""

from dataclasses import dataclass, replace

import dolfinx.fem
import dolfinx.fem.petsc
import numpy as np
import qugar.dolfinx
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from cutfem import core
from cutfem.problems import dfg2d1

# DFG 2D-3 reference values (FEATFLOW, Q2/P1disc level 6, Crank-Nicolson,
# dt = 1/1600).
DFG_2D3_REF = {"c_D_max": 2.9437637, "t_c_D_max": 3.9365625,
               "c_L_max": 0.4774878, "t_c_L_max": 5.6928125}

# BDF coefficients (a0, a1, a2) of (a0 u^{n+1} + a1 u^n + a2 u^{n-1}) / dt
BDF1 = (1.0, -1.0, 0.0)
BDF2 = (1.5, -2.0, 0.5)


@dataclass(frozen=True)
class Params(dfg2d1.Params):
    """Parameters of one DFG 2D-3 run (geometry and penalties as in 2D-1)."""

    U_m: float = 1.5            # amplitude of the maximum inflow velocity
    U: float = 1.0              # peak mean velocity (ghost penalty weight)
    tol: float = 1e-8           # Newton tolerance on the residual norm
    max_iter: int = 10          # Newton iterations per step
    dt: float = 1.0 / 1600.0
    t_end: float = 8.0


@dataclass
class System(core.System):
    """DFG 2D-3 system: residual and Jacobian forms, time levels, solver."""

    params: Params
    V: object
    Q: object
    measures: core.CutMeasures
    consts: dict        # fem.Constants: nu, gamma_*, phi, dt, bdf (a0, a1, a2)
    uh: object          # current iterate u^{n+1}
    ph: object
    u_n: object         # u^n
    u_nm1: object       # u^{n-1}
    g: object           # Dirichlet data (inflow and walls) at the current time
    profile: np.ndarray  # Dirichlet data for a unit inflow amplitude
    F_forms: list       # [F_u, F_p]: residual without ghost penalty
    J_forms: list       # Jacobian blocks of F_forms
    force_forms: list   # x and y components of the force on the cylinder
    bcs: list
    G: PETSc.Mat        # ghost penalty matrix (linear, assembled once)
    w: PETSc.Vec        # monolithic (u, p) iterate
    dw: PETSc.Vec       # Newton increment
    ksp: PETSc.KSP

    def destroy(self):
        """Destroy the PETSc objects, including the solver."""
        super().destroy()
        self.ksp.destroy()


def phi(p: Params, h):
    """Ghost penalty weight of 2D-1 plus h^2 / dt for the mass term."""
    return dfg2d1._phi(p.nu, p.U, h) + h ** 2 / p.dt


def inflow_amplitude(p: Params, t):
    """Maximum inflow velocity U_m sin(pi t / 8)."""
    return p.U_m * np.sin(np.pi * t / 8.0)


def build(p: Params) -> System:
    """Build residual, Jacobian, ghost penalty matrix and solver."""
    mesh, h = dfg2d1.channel_mesh(p)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.v_degree, shape=(2,))
    Q = core.lagrange_space(mesh, p.p_degree)
    du, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dp, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    uh, u_n, u_nm1 = (dolfinx.fem.Function(V, name=name)
                      for name in ("uh", "u_n", "u_nm1"))
    ph = dolfinx.fem.Function(Q, name="ph")

    def constant(value):
        return dolfinx.fem.Constant(mesh, core.DTYPE(value))

    consts = {key: constant(getattr(p, key))
              for key in ("nu", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p", "dt")}
    consts["phi"] = constant(phi(p, h))
    consts["bdf"] = dolfinx.fem.Constant(mesh, np.array(BDF1, dtype=core.DTYPE))
    nu, gN1, gN2, dt = (consts[k] for k in ("nu", "gamma_N1", "gamma_N2", "dt"))
    a = consts["bdf"]
    n, dx, ds = m.n, m.dx, m.ds

    # Residual; Nitsche no-slip on the cylinder (n points out of the fluid)
    u_t = (a[0] * uh + a[1] * u_n + a[2] * u_nm1) / dt
    F_u = (ufl.inner(u_t, v) * dx
           + nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * dx
           + ufl.inner(ufl.dot(ufl.grad(uh), uh), v) * dx
           - ph * ufl.div(v) * dx + ph * ufl.dot(v, n) * ds
           - nu * ufl.inner(ufl.dot(ufl.grad(uh), n), v) * ds
           - nu * ufl.inner(ufl.dot(ufl.grad(v), n), uh) * ds
           + (gN1 * nu / h) * ufl.inner(uh, v) * ds
           + (gN2 / h) * ufl.dot(uh, n) * ufl.dot(v, n) * ds)
    F_p = ufl.div(uh) * q * dx - q * ufl.dot(uh, n) * ds
    J = [[ufl.derivative(F_u, uh, du), ufl.derivative(F_u, ph, dp)],
         [ufl.derivative(F_p, uh, du), None]]
    F_forms = [core.compile_form(F_u), core.compile_form(F_p)]
    J_forms = [[core.compile_form(J_ij) if J_ij is not None else None
                for J_ij in row] for row in J]

    G_uu = consts["gamma_u"] * consts["phi"] * core.ghost_penalty(du, v, m, h, order=2)
    G_pp = (consts["gamma_p"] / consts["phi"] * h ** 2
            * core.ghost_penalty(dp, q, m, h, order=1))
    g_forms = [[core.compile_form(G_uu, cut=False), None],
               [None, core.compile_form(G_pp, cut=False)]]

    # Nitsche-consistent traction, as in dfg2d1.benchmark
    traction = (ph * n - nu * ufl.dot(ufl.grad(uh), n)
                + (gN1 * nu / h) * uh + (gN2 / h) * ufl.dot(uh, n) * n)
    force_forms = [qugar.dolfinx.form_custom(traction[i] * ds) for i in range(2)]

    bcs = dfg2d1._dirichlet_bcs(mesh, V, replace(p, U_m=1.0))
    g = bcs[0].g
    profile = g.x.array.copy()
    g.x.array[:] = 0.0

    # Ghost penalty is linear: assemble once. Allocate the Jacobian with the
    # union of both sparsity patterns, as in dfg2d1.
    G = dolfinx.fem.petsc.create_matrix(g_forms)
    dolfinx.fem.petsc.assemble_matrix(G, g_forms, bcs=bcs, diag=0.0)
    G.assemble()
    A = dolfinx.fem.petsc.create_matrix(J_forms)
    dolfinx.fem.petsc.assemble_matrix(A, J_forms, bcs=bcs)
    A.assemble()
    A.axpy(1.0, G, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    A.zeroEntries()
    b = dolfinx.fem.petsc.create_vector(
        dolfinx.fem.extract_function_spaces(F_forms), kind=PETSc.Vec.Type.MPI)

    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    return System(mesh=mesh, h=h, A=A, b=b, params=p, V=V, Q=Q, measures=m,
                  consts=consts, uh=uh, ph=ph, u_n=u_n, u_nm1=u_nm1, g=g,
                  profile=profile, F_forms=F_forms, J_forms=J_forms,
                  force_forms=force_forms, bcs=bcs, G=G,
                  w=A.createVecRight(), dw=A.createVecRight(), ksp=ksp)


def _assemble_residual(s: System):
    """Assemble b = F(w) + G w, zero on the Dirichlet rows (w satisfies BCs)."""
    with s.b.localForm() as local:
        local.set(0.0)
    dolfinx.fem.petsc.assemble_vector(s.b, s.F_forms)
    s.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    s.G.multAdd(s.w, s.b, s.b)
    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(s.F_forms), s.bcs)
    dolfinx.fem.petsc.set_bc(s.b, bcs0, x0=s.w, alpha=-1.0)


def _assemble_jacobian(s: System):
    """Assemble A = J(w) + G, identity on the Dirichlet rows."""
    s.A.zeroEntries()
    dolfinx.fem.petsc.assemble_matrix(s.A, s.J_forms, bcs=s.bcs)
    s.A.assemble()
    s.A.axpy(1.0, s.G, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)


def newton(s: System):
    """Newton iteration J(w) dw = F(w), w <- w - dw; returns the residual norms.

    The iterate must satisfy the Dirichlet conditions on entry. Raises
    RuntimeError if ||F|| < tol is not reached in ``max_iter`` iterations.
    """
    p = s.params
    core.join_solution(s.w, s.uh, s.ph)
    history = []
    for _ in range(p.max_iter + 1):
        _assemble_residual(s)
        history.append(s.b.norm())
        if history[-1] < p.tol:
            return history
        if len(history) > p.max_iter:
            break
        _assemble_jacobian(s)
        s.ksp.setOperators(s.A)
        s.ksp.solve(s.b, s.dw)
        if s.ksp.getConvergedReason() < 0:
            raise RuntimeError(f"linear solve failed (KSP reason "
                               f"{s.ksp.getConvergedReason()})")
        s.w.axpy(-1.0, s.dw)
        core.split_solution(s.w, s.uh, s.ph)
    raise RuntimeError(f"Newton did not converge: ||F|| = {history}")


def step(s: System, t, bdf):
    """Advance from t - dt to t with the given BDF coefficients."""
    s.consts["bdf"].value[:] = bdf
    # Initial guess: extrapolation 2 u^n - u^{n-1} (u^n on the first step)
    if bdf == BDF2:
        s.uh.x.array[:] = 2.0 * s.u_n.x.array - s.u_nm1.x.array
    else:
        s.uh.x.array[:] = s.u_n.x.array
    s.g.x.array[:] = inflow_amplitude(s.params, t) * s.profile
    for bc in s.bcs:
        bc.set(s.uh.x.array)
    s.uh.x.scatter_forward()

    history = newton(s)
    s.u_nm1.x.array[:] = s.u_n.x.array
    s.u_n.x.array[:] = s.uh.x.array
    return history


def drag_lift(s: System):
    """c_D and c_L of the current solution, normalized with U_mean = 2/3 U_m."""
    p = s.params
    force = [s.mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(f, coeffs=f.pack_coefficients()), op=MPI.SUM)
        for f in s.force_forms]
    scale = 2.0 / ((2.0 / 3.0 * p.U_m) ** 2 * 2.0 * p.radius)
    return scale * force[0], scale * force[1]


def solve(s: System, on_step=None):
    """Time loop on [0, t_end] from rest; returns the force history.

    ``on_step(n, record)`` is called at the start (n = 0) and after every
    step with the step number and the record so far (dict of lists: t, c_D,
    c_L, newton).
    """
    p = s.params
    n_steps = round(p.t_end / p.dt)
    record = {"t": [0.0], "c_D": [0.0], "c_L": [0.0], "newton": [0]}
    if on_step is not None:
        on_step(0, record)
    for k in range(1, n_steps + 1):
        t = k * p.dt
        history = step(s, t, BDF1 if k == 1 else BDF2)
        c_D, c_L = drag_lift(s)
        for key, value in zip(record, (t, c_D, c_L, len(history) - 1),
                              strict=True):
            record[key].append(value)
        if on_step is not None:
            on_step(k, record)
    return record


def peaks(record) -> dict:
    """Maximum c_D and c_L and their times."""
    i_D, i_L = int(np.argmax(record["c_D"])), int(np.argmax(record["c_L"]))
    return {"c_D_max": record["c_D"][i_D], "t_c_D_max": record["t"][i_D],
            "c_L_max": record["c_L"][i_L], "t_c_L_max": record["t"][i_L]}


def vtk_series(path, s: System, degree=2) -> core.VTKSeries:
    """ParaView time series of velocity and pressure (degree 2 is exact for u_h)."""
    return core.VTKSeries(path, s.mesh, {"uh": s.uh, "ph": s.ph}, degree=degree)


def load_reference(path):
    """FEATFLOW bdforces file: columns step, t, boundary, c_D, c_L."""
    data = np.loadtxt(path, comments="#")
    return {"t": data[:, 1], "c_D": data[:, 3], "c_L": data[:, 4]}
