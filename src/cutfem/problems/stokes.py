"""Stokes problem on a disk, Nitsche boundary conditions and ghost penalty.

    -mu Delta u + grad p = f,   div u = 0   in Omega,   u = u_ex on Gamma,

with u_ex = (20 x y^3, 5 x^4 - 5 y^4) and p_ex = 60 x^2 y - 20 y^3.
"""

from dataclasses import dataclass

import dolfinx.fem
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from cutfem import core

ERRORS = ("L2_u", "H1_u", "L2_p", "div_u")


@dataclass(frozen=True)
class Params:
    """Parameters of one Stokes run."""

    n_cells: int = 32
    gamma_mu: float = 10.0      # Nitsche penalty
    gamma_g: float = 0.1        # ghost penalty (velocity and pressure)
    gamma_p: float = 0.0        # global pressure stabilization (CIP)
    mu: float = 1.0             # viscosity
    v_degree: int = 2
    p_degree: int = 1
    center: tuple = (0.0, 0.0)
    radius: float = 0.5
    xmin: tuple = (-1.0, -1.0)
    xmax: tuple = (1.0, 1.0)


@dataclass
class System(core.System):
    """Assembled Stokes system; ``A`` is the monolithic matrix, unpinned."""

    params: Params
    V: object
    Q: object
    measures: core.CutMeasures
    A_uu: PETSc.Mat     # velocity block, incl. ghost penalty
    A_up: PETSc.Mat     # b_h(p, v)
    A_pu: PETSc.Mat     # -b_h(q, u)
    A_pp: PETSc.Mat     # pressure stabilization (ghost penalty + CIP)


def exact_velocity(x):
    """Exact velocity as a UFL expression of the spatial coordinates."""
    return ufl.as_vector([20 * x[0] * x[1] ** 3, 5 * x[0] ** 4 - 5 * x[1] ** 4])


def exact_pressure(x):
    """Exact pressure; works for UFL coordinates and NumPy arrays."""
    return 60 * x[0] ** 2 * x[1] - 20 * x[1] ** 3


def build(p: Params) -> System:
    """Assemble the Stokes CutFEM system (blocks and monolithic matrix)."""
    mesh, h = core.disk_mesh(p.n_cells, p.xmin, p.xmax, p.center, p.radius)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.v_degree, shape=(2,))
    Q = core.lagrange_space(mesh, p.p_degree)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    pr, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    x = ufl.SpatialCoordinate(mesh)
    g = exact_velocity(x)
    f = -p.mu * ufl.div(ufl.grad(g)) + ufl.grad(exact_pressure(x))
    mu, n = p.mu, m.n
    zero = dolfinx.fem.Constant(mesh, core.DTYPE(0.0))

    # Velocity block: viscous term with Nitsche, ghost penalty up to order k
    a_uu = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * m.dx
            - mu * ufl.inner(ufl.dot(ufl.grad(u), n), v) * m.ds
            - mu * ufl.inner(u, ufl.dot(ufl.grad(v), n)) * m.ds
            + (p.gamma_mu * mu / h) * ufl.inner(u, v) * m.ds)
    A_uu = core.assemble_matrix(a_uu)
    if p.gamma_g != 0.0:
        core.add_matrix(A_uu, p.gamma_g * mu
                        * core.ghost_penalty(u, v, m, h, order=p.v_degree))

    # Coupling blocks
    A_up = core.assemble_matrix(-pr * ufl.div(v) * m.dx
                                + pr * ufl.dot(v, n) * m.ds)
    A_pu = core.assemble_matrix(ufl.div(u) * q * m.dx
                                - q * ufl.dot(u, n) * m.ds)

    # Pressure block: ghost penalty (h^(2j+1) scaling) and global CIP
    A_pp = core.assemble_matrix(zero * pr * q * m.dx)
    if p.gamma_g != 0.0:
        core.add_matrix(A_pp, (p.gamma_g / mu) * h ** 2
                        * core.ghost_penalty(pr, q, m, h, order=p.p_degree))
    if p.gamma_p != 0.0:
        jumps = ufl.inner(core.normal_jump(pr, m.n_F), core.normal_jump(q, m.n_F))
        core.add_matrix(A_pp, (p.gamma_p * h ** 3 / mu) * jumps * m.dS)

    nest = PETSc.Mat().createNest([[A_uu, A_up], [A_pu, A_pp]])
    nest.assemble()
    A = PETSc.Mat()
    nest.convert("aij", A)
    A.assemble()
    nest.destroy()

    # Right-hand side, in the same [u, p] block layout as A
    b_u = core.assemble_vector(
        ufl.inner(f, v) * m.dx
        - mu * ufl.inner(g, ufl.dot(ufl.grad(v), n)) * m.ds
        + (p.gamma_mu * mu / h) * ufl.inner(g, v) * m.ds)
    b_p = core.assemble_vector(zero * q * m.dx - q * ufl.dot(g, n) * m.ds)
    b = A.createVecRight()
    n_u = b_u.getLocalSize()
    b.array[:n_u] = b_u.array_r
    b.array[n_u:] = b_p.array_r
    b_u.destroy()
    b_p.destroy()

    return System(mesh=mesh, h=h, A=A, b=b, params=p, V=V, Q=Q, measures=m,
                  A_uu=A_uu, A_up=A_up, A_pu=A_pu, A_pp=A_pp)


def _pressure_pin(s: System):
    """Global row and exact value of one pressure unknown in a full cell.

    The pin is placed by the lowest rank owning a full cell; other ranks
    return ``(None, None)``.
    """
    comm = s.mesh.comm
    n_owned = s.Q.dofmap.index_map.size_local
    dof = None
    for cell in s.mesh.get_full_cells():
        owned = [d for d in s.Q.dofmap.cell_dofs(cell) if d < n_owned]
        if owned:
            dof = owned[0]
            break
    pin_rank = comm.allreduce(comm.rank if dof is not None else comm.size,
                              op=MPI.MIN)
    if pin_rank == comm.size:
        raise RuntimeError("no full cell to pin the pressure")
    if comm.rank != pin_rank:
        return None, None

    n_u_owned = s.V.dofmap.index_map.size_local * s.V.dofmap.index_map_bs
    row = s.A.getOwnershipRange()[0] + n_u_owned + dof
    value = float(exact_pressure(s.Q.tabulate_dof_coordinates()[dof]))
    return row, value


def solve(s: System):
    """Solve with one pinned pressure unknown; returns ``(uh, ph)``.

    The pin (pressure defined up to a constant) is applied to copies, so
    ``s.A`` and ``s.b`` stay the discrete Stokes system.
    """
    row, value = _pressure_pin(s)
    rows = [] if row is None else [row]
    A, b = s.A.copy(), s.b.copy()
    A.zeroRows(rows, diag=1.0)
    b.setValues(rows, [] if row is None else [value])
    b.assemble()

    x = core.lu_solve(A, b)
    uh = dolfinx.fem.Function(s.V, name="uh")
    ph = dolfinx.fem.Function(s.Q, name="ph")
    core.split_solution(x, uh, ph)
    for obj in (A, b, x):
        obj.destroy()
    return uh, ph


def errors(s: System, solution) -> dict:
    """Velocity errors, mean-free pressure error and divergence residual."""
    uh, ph = solution
    dx = s.measures.dx
    x = ufl.SpatialCoordinate(s.mesh)
    u_ex, p_ex = exact_velocity(x), exact_pressure(x)

    area = core.integrate(dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0)) * dx)
    mean_h = core.integrate(ph * dx) / area
    mean_ex = core.integrate(p_ex * dx) / area
    e_u = uh - u_ex
    e_p = (ph - mean_h) - (p_ex - mean_ex)
    return {"L2_u": core.l2_norm(e_u, dx),
            "H1_u": core.h1_seminorm(e_u, dx),
            "L2_p": core.l2_norm(e_p, dx),
            "div_u": core.l2_norm(ufl.div(uh), dx)}


def expected_rates(p: Params) -> dict:
    """Optimal orders for degrees (k, l) = (v_degree, p_degree)."""
    return {"L2_u": p.v_degree + 1, "H1_u": p.v_degree,
            "L2_p": p.p_degree + 1, "div_u": p.v_degree}


def inf_sup(s: System, max_dense=1e8, zero_tol=1e-10) -> dict:
    """Numerical inf-sup constant (Chapelle & Bathe test), serial only.

    Solves (B N^-1 B^T + G_p) q = lambda M_T q on pressures orthogonal to the
    constants, with
        B    matrix of b_h(q, v) (incl. the Nitsche boundary term),
        N    velocity norm mu ||grad v||^2_{Omega_T} + mu h^-1 ||v||^2_Gamma,
        G_p  pressure stabilization (``s.A_pp``),
        M_T  pressure mass matrix on Omega_T (weight 1/mu).

    Returns
    -------
    dict with ``beta`` (sqrt of lambda_min), ``lam_min``, ``lam_max``,
    ``kappa`` (lambda_max / lambda_min), ``n_small`` (eigenvalues below
    ``zero_tol * lam_max``), ``rq_const`` (Rayleigh quotient of the constant
    pressure, should be ~0) and ``skew`` (||A_pu + A_up^T|| / ||A_up||,
    should be ~0). NaN entries if the dense problem exceeds ``max_dense``.
    """
    import scipy.linalg as sla
    import scipy.sparse.linalg as spla

    if s.mesh.comm.size > 1:
        raise RuntimeError("inf_sup is serial only")

    B = core.to_scipy(s.A_up).T.tocsr()
    n_Q, n_V = B.shape
    keys = ("beta", "lam_min", "lam_max", "kappa", "n_small", "rq_const", "skew")
    if n_Q * n_V > max_dense:
        return dict.fromkeys(keys, np.nan)

    mu, h, m = s.params.mu, s.h, s.measures
    u, v = ufl.TrialFunction(s.V), ufl.TestFunction(s.V)
    pr, q = ufl.TrialFunction(s.Q), ufl.TestFunction(s.Q)
    N_V = core.assemble_matrix(mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * m.dx,
                               cut=False)
    core.add_matrix(N_V, (mu / h) * ufl.inner(u, v) * m.ds, cut=True)
    M_T = core.assemble_matrix((1.0 / mu) * pr * q * m.dx, cut=False)

    skew = spla.norm(core.to_scipy(s.A_pu) + B) / spla.norm(B)
    X = spla.splu(core.to_scipy(N_V).tocsc()).solve(B.T.toarray())
    K = B @ X + core.to_scipy(s.A_pp).toarray()
    K = 0.5 * (K + K.T)
    M = core.to_scipy(M_T).toarray()
    M = 0.5 * (M + M.T)
    N_V.destroy()
    M_T.destroy()

    ones = np.ones(n_Q)
    rq_const = float(ones @ K @ ones) / float(ones @ M @ ones)
    Z = sla.null_space((M @ ones)[None, :])
    lam = sla.eigh(Z.T @ K @ Z, Z.T @ M @ Z, eigvals_only=True)

    lam_min, lam_max = float(lam[0]), float(lam[-1])
    return {"beta": np.sqrt(lam_min) if lam_min > 0 else 0.0,
            "lam_min": lam_min,
            "lam_max": lam_max,
            "kappa": lam_max / lam_min if lam_min > 0 else np.inf,
            "n_small": int(np.sum(lam < zero_tol * lam_max)),
            "rq_const": rq_const,
            "skew": float(skew)}


def save_vtk(path, s: System, solution) -> None:
    """Write velocity and pressure with exact fields and errors."""
    uh, ph = solution
    core.save_vtk(path, s.mesh, {"uh": uh, "ph": ph},
                  exact={"uh": exact_velocity, "ph": exact_pressure})
