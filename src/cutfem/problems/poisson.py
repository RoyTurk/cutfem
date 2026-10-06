"""Poisson problem on a disk, Nitsche boundary conditions and ghost penalty.

    -Delta u = f in Omega,    u = u_ex on Gamma,

with u_ex = sin(pi x) sin(pi y).
"""

from dataclasses import dataclass

import dolfinx.fem
import numpy as np
import ufl

from cutfem import core

ERRORS = ("L2_u", "H1_u")


@dataclass(frozen=True)
class Params:
    """Parameters of one Poisson run."""

    n_cells: int = 32
    beta: float = 10.0          # Nitsche penalty
    tau: float = 1.0            # ghost penalty
    degree: int = 1
    center: tuple = (0.0, 0.0)
    radius: float = 0.5
    xmin: tuple = (-1.0, -1.0)
    xmax: tuple = (1.0, 1.0)


@dataclass
class System(core.System):
    """Assembled Poisson system."""

    params: Params
    V: object
    dx: object
    u_exact: object


def exact_solution(x):
    """Exact solution as a UFL expression of the spatial coordinates."""
    return ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])


def build(p: Params) -> System:
    """Assemble the Poisson CutFEM system."""
    mesh, h = core.disk_mesh(p.n_cells, p.xmin, p.xmax, p.center, p.radius)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.degree)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    u_ex = exact_solution(ufl.SpatialCoordinate(mesh))
    f = -ufl.div(ufl.grad(u_ex))
    du_n = ufl.dot(ufl.grad(u), m.n)
    dv_n = ufl.dot(ufl.grad(v), m.n)

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * m.dx
         - du_n * v * m.ds - u * dv_n * m.ds + (p.beta / h) * u * v * m.ds)
    L = (f * v * m.dx
         + (p.beta / h) * u_ex * v * m.ds - u_ex * dv_n * m.ds)

    A = core.assemble_matrix(a)
    if p.tau != 0.0:
        core.add_matrix(A, p.tau * core.ghost_penalty(u, v, m, h, order=1))
    b = core.assemble_vector(L)

    return System(mesh=mesh, h=h, A=A, b=b, params=p, V=V, dx=m.dx,
                  u_exact=u_ex)


def solve(s: System):
    """Solve the system; returns the discrete solution ``uh``."""
    x = core.lu_solve(s.A, s.b)
    uh = dolfinx.fem.Function(s.V, name="uh")
    core.split_solution(x, uh)
    x.destroy()
    return uh


def errors(s: System, uh) -> dict:
    """L2 and H1-seminorm errors over the physical domain."""
    e = uh - s.u_exact
    return {"L2_u": core.l2_norm(e, s.dx), "H1_u": core.h1_seminorm(e, s.dx)}


def expected_rates(p: Params) -> dict:
    """Optimal convergence orders for Lagrange elements of degree k."""
    return {"L2_u": p.degree + 1, "H1_u": p.degree}


def save_vtk(path, s: System, uh) -> None:
    """Write uh, the exact solution and the error for ParaView."""
    core.save_vtk(path, s.mesh, {"uh": uh}, exact={"uh": exact_solution})
