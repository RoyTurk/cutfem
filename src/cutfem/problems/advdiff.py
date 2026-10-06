"""Advection-diffusion around a disk with Nitsche conditions and CIP.

    -mu Delta u + beta . grad u = 0   in Omega = box minus disk,
    u = g on the disk boundary Gamma (Nitsche),   u = 0 on the box boundary.

Stabilization: ghost penalty on the cut facets (viscous and convective
weight) and continuous interior penalty (CIP) on all interior facets.
There is no exact solution: ``errors`` is empty, studies use line profiles.
"""

from dataclasses import dataclass

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import ufl
from petsc4py import PETSc

from cutfem import core

ERRORS = ()


@dataclass(frozen=True)
class Params:
    """Parameters of one advection-diffusion run."""

    n_cells: int = 64
    mu: float = 5e-3            # diffusion
    beta: tuple = (1.0, 0.0)    # advection field
    gamma: float = 10.0         # Nitsche penalty
    gamma_mu: float = 0.1       # viscous ghost penalty
    gamma_beta: float = 0.5     # CIP and convective ghost penalty
    g: float = 1.0              # Dirichlet value on the disk
    degree: int = 1
    center: tuple = (0.2, 0.5)
    radius: float = 0.1
    xmin: tuple = (0.0, 0.0)
    xmax: tuple = (1.0, 1.0)


@dataclass
class System(core.System):
    """Assembled advection-diffusion system."""

    params: Params
    V: object


def build(p: Params) -> System:
    """Assemble the advection-diffusion CutFEM system."""
    mesh, h = core.disk_mesh(p.n_cells, p.xmin, p.xmax, p.center, p.radius,
                             inside=False)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.degree)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    mu, n = p.mu, m.n
    beta_norm = float(np.linalg.norm(p.beta))
    beta = dolfinx.fem.Constant(mesh, np.asarray(p.beta, dtype=core.DTYPE))
    g = dolfinx.fem.Constant(mesh, core.DTYPE(p.g))
    f = dolfinx.fem.Constant(mesh, core.DTYPE(0.0))
    beta_n = ufl.dot(beta, n)
    inflow = 0.5 * (beta_n - abs(beta_n))

    a = (mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * m.dx
         - mu * ufl.inner(ufl.grad(u), n) * v * m.ds
         - mu * ufl.inner(ufl.grad(v), n) * u * m.ds
         + (p.gamma * mu / h) * u * v * m.ds
         + ufl.dot(beta, ufl.grad(u)) * v * m.dx
         - inflow * u * v * m.ds)
    L = (f * v * m.dx
         - mu * g * ufl.inner(ufl.grad(v), n) * m.ds
         + (p.gamma * mu / h) * g * v * m.ds
         - inflow * g * v * m.ds)

    # CIP on all interior facets, ghost penalty on the cut facets
    phi = beta_norm ** 2 * h ** 2 / (mu + beta_norm * h)
    jumps = ufl.inner(core.normal_jump(u, m.n_F), core.normal_jump(v, m.n_F))
    stab = (p.gamma_beta * phi * h * jumps * m.dS
            + (p.gamma_mu * mu + p.gamma_beta * phi) * h * jumps * m.dS_ghost)

    bcs = [dolfinx.fem.dirichletbc(core.DTYPE(0.0), _box_boundary_dofs(mesh, V, p), V)]
    a_form = core.compile_form(a)
    stab_form = core.compile_form(stab, cut=False)
    L_form = core.compile_form(L)

    A = dolfinx.fem.petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    core.add_matrix(A, stab, cut=False, bcs=bcs)

    b = dolfinx.fem.petsc.assemble_vector(L_form)
    dolfinx.fem.petsc.apply_lifting(b, [a_form], bcs=[bcs])
    dolfinx.fem.petsc.apply_lifting(b, [stab_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)

    return System(mesh=mesh, h=h, A=A, b=b, params=p, V=V)


def _box_boundary_dofs(mesh, V, p: Params):
    """DoFs on the boundary of the background box."""
    fdim = mesh.topology.dim - 1
    xmin, xmax = p.xmin, p.xmax
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim,
        lambda x: (np.isclose(x[0], xmin[0]) | np.isclose(x[0], xmax[0])
                   | np.isclose(x[1], xmin[1]) | np.isclose(x[1], xmax[1])))
    return dolfinx.fem.locate_dofs_topological(V, fdim, facets)


def solve(s: System):
    """Solve the system; returns the discrete solution ``uh``."""
    x = core.lu_solve(s.A, s.b)
    uh = dolfinx.fem.Function(s.V, name="uh")
    core.split_solution(x, uh)
    x.destroy()
    return uh


def errors(s: System, uh) -> dict:
    """No exact solution: no error norms."""
    return {}


def expected_rates(p: Params) -> dict:
    """No exact solution: no expected orders."""
    return {}


def peclet_h(p: Params, h: float) -> float:
    """Grid Peclet number |beta| h / (2 mu)."""
    return float(np.linalg.norm(p.beta)) * h / (2.0 * p.mu)


def disk_chord(p: Params, y: float):
    """Interval (x0, x1) of the line at height y inside the disk, or None."""
    half = np.sqrt(max(p.radius ** 2 - (y - p.center[1]) ** 2, 0.0))
    return (p.center[0] - half, p.center[0] + half) if half > 0 else None


def sample_line(s: System, uh, x, y):
    """Values of ``uh`` at the points (x, y); NaN inside the disk."""
    x = np.asarray(x, dtype=float)
    values = core.eval_at_points(uh, np.column_stack([x, np.full_like(x, y)]))
    cx, cy = s.params.center
    values[np.hypot(x - cx, y - cy) < s.params.radius] = np.nan
    return values


def save_vtk(path, s: System, uh) -> None:
    """Write the solution for ParaView."""
    core.save_vtk(path, s.mesh, {"uh": uh})
