"""Stationary Navier-Stokes flow through a square duct filled with a gyroid.

    -nu Delta u + (u . grad) u + grad p = 0,   div u = 0   in Omega,

    Omega = {x in B : phi(x) < c},   B = (0, n_x) x (0, 1) x (0, 1),
    phi   = sin X cos Y + sin Y cos Z + sin Z cos X,   X = 2 pi x, ...

Lengths are scaled by the unit cell, velocities by the mean inflow speed U,
so nu = U / Re. Boundary conditions:

* inlet x = 0: u = g = (36 U y z (1 - y)(1 - z), 0, 0), strong;
* side faces y, z in {0, 1}: u = 0, strong (g vanishes there, so a single
  Dirichlet condition with g covers inlet and walls);
* outlet x = n_x: do-nothing (natural);
* gyroid wall phi = c: u = 0 by Nitsche, as the cylinder in dfg2d1.

Same discretization as :mod:`cutfem.problems.dfg2d1` (Taylor-Hood, ghost
penalty, Picard iteration), whose assembly and solver are reused.
"""

from dataclasses import dataclass

import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import qugar.impl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from cutfem import core
from cutfem.problems import dfg2d1

TAU = 2.0 * np.pi
QUANTITIES = ("Q_in", "Q_mean", "dp_cell", "k_app", "div_L2")


@dataclass(frozen=True)
class Params:
    """Parameters of one gyroid duct run."""

    n_x: int = 4                # unit cells along the duct
    n_per_cell: int = 8         # background cells per unit cell and direction
    c: float = 0.6              # level: fluid = phi < c
    Re: float = 1.0             # U * (cell size) / nu
    U: float = 1.0              # mean inflow speed over the full inlet face
    gamma_u: float = 0.1        # velocity ghost penalty
    gamma_p: float = 0.1        # pressure ghost penalty
    gamma_N1: float = 35.0      # Nitsche viscous penalty
    gamma_N2: float = 35.0      # Nitsche normal penalty
    v_degree: int = 2
    p_degree: int = 1
    tol: float = 1e-10          # Picard tolerance on the residual norm
    max_iter: int = 50

    @property
    def nu(self):
        """Kinematic viscosity U / Re (unit cell size)."""
        return self.U / self.Re

    @property
    def xmax(self):
        return np.array([float(self.n_x), 1.0, 1.0])


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
    bcs: list           # strong Dirichlet BCs (inlet and side walls)
    G: PETSc.Mat        # ghost penalty matrix


# Geometry


def gyroid(x, y, z):
    """The level-set function of qugar.impl.create_Schoen([1, 1, 1])."""
    X, Y, Z = TAU * x, TAU * y, TAU * z
    return np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)


def is_fluid(points, c):
    """True where phi < c; points of shape (n, 3)."""
    return gyroid(*np.asarray(points).T) < c


def duct_mesh(p: Params, comm=MPI.COMM_WORLD):
    """Unfitted Cartesian mesh of the duct, fluid = {phi < c}; ``(mesh, h)``."""
    func = qugar.impl.create_functions_addition(
        qugar.impl.create_Schoen(periods=[1.0, 1.0, 1.0]),
        qugar.impl.create_constant(-p.c, 3, use_bzr=False))
    n_cells = [p.n_x * p.n_per_cell, p.n_per_cell, p.n_per_cell]
    # Ghost cells across rank boundaries for the ghost penalty, as disk_mesh
    mesh = create_unfitted_impl_Cartesian_mesh(
        comm, func, n_cells, np.zeros(3, dtype=core.DTYPE),
        p.xmax.astype(core.DTYPE), exclude_empty_cells=True,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet, dtype=core.DTYPE)
    return mesh, 1.0 / p.n_per_cell


def inflow(x, U=1.0):
    """g_x = 36 U y z (1 - y)(1 - z): mean U over the face, 0 on its edges."""
    return 36.0 * U * x[1] * x[2] * (1.0 - x[1]) * (1.0 - x[2])


def _dirichlet_bcs(mesh, V, p: Params):
    """u = g on the inlet and the side faces (g = 0 on the side faces)."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    facets = dolfinx.mesh.locate_entities_boundary(
        mesh, fdim,
        lambda x: (np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)
                   | np.isclose(x[1], 1.0) | np.isclose(x[2], 0.0)
                   | np.isclose(x[2], 1.0)))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)

    g = dolfinx.fem.Function(V, name="g_D")
    g.interpolate(lambda x: np.vstack((inflow(x, p.U), np.zeros_like(x[0]),
                                       np.zeros_like(x[0]))))
    return [dolfinx.fem.dirichletbc(g, dofs)]


# System


def build(p: Params) -> System:
    """Build forms, matrices and Picard state for the gyroid duct."""
    mesh, h = duct_mesh(p)
    m = core.cut_measures(mesh)
    V = core.lagrange_space(mesh, p.v_degree, shape=(3,))
    Q = core.lagrange_space(mesh, p.p_degree)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    pr, q = ufl.TrialFunction(Q), ufl.TestFunction(Q)

    def constant(value):
        return dolfinx.fem.Constant(mesh, core.DTYPE(value))

    consts = {key: constant(getattr(p, key))
              for key in ("nu", "gamma_N1", "gamma_N2", "gamma_u", "gamma_p")}
    consts["phi"] = constant(dfg2d1._phi(p.nu, p.U, h))
    nu, gN1, gN2 = consts["nu"], consts["gamma_N1"], consts["gamma_N2"]
    n, dx, ds = m.n, m.dx, m.ds
    zero = dolfinx.fem.Constant(mesh, np.zeros(3, dtype=core.DTYPE))
    u_k = dolfinx.fem.Function(V, name="u_k")

    # Oseen operator at u_k, Nitsche on the gyroid (n points out of the fluid)
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

    # No body force and a homogeneous wall condition: zero right-hand sides
    L_forms = [core.compile_form(ufl.inner(zero, v) * dx),
               core.compile_form(constant(0.0) * q * dx)]

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


def solve(s: System, max_iter=None, verbose=False):
    """Picard iteration of dfg2d1; returns ``(uh, ph, history)``."""
    return dfg2d1.solve(s, max_iter=max_iter, verbose=verbose)


# Quantities of interest


def inlet_flux(p: Params, n=1000):
    """Q_in = int_{Gamma_in} g_x dA (fluid part of the inlet face), midpoint rule."""
    s = (np.arange(n) + 0.5) / n
    y, z = np.meshgrid(s, s, indexing="ij")
    points = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
    g = inflow(points.T, p.U)
    return float(np.mean(g * is_fluid(points, p.c)))


def cell_averages(s: System, solution):
    """Exact (qugar quadrature) averages over each unit cell [i, i + 1].

    The cell boundaries are mesh planes, so the indicator of a cell is
    constant on every background cell and the integrals are exact up to
    quadrature. Returns per-cell fluid volume, mean u_x flux (by Gauss,
    int u_x dV / 1 = mean of Q(x) over the cell) and mean pressure.
    """
    uh, ph, _ = solution
    x = ufl.SpatialCoordinate(s.mesh)
    dx = s.measures.dx
    volume, flux, pressure = [], [], []
    for i in range(s.params.n_x):
        chi = ufl.conditional(ufl.And(ufl.gt(x[0], i), ufl.lt(x[0], i + 1)), 1.0, 0.0)
        vol = core.integrate(chi * dx)
        volume.append(vol)
        flux.append(core.integrate(chi * uh[0] * dx))
        pressure.append(core.integrate(chi * ph * dx) / vol)
    return {"volume": np.array(volume), "flux": np.array(flux),
            "pressure": np.array(pressure)}


def slice_points(p: Params, axis, value, n):
    """Grid of n x n (or n_x n x n) points on the plane x_axis = value."""
    extent = p.xmax
    axes = [k for k in range(3) if k != axis]
    counts = [round(n * extent[k]) for k in axes]
    grids = [(np.arange(m) + 0.5) / m * extent[k] for m, k in zip(counts, axes)]
    a, b = np.meshgrid(*grids, indexing="ij")
    points = np.empty((a.size, 3))
    points[:, axis] = value
    points[:, axes[0]], points[:, axes[1]] = a.ravel(), b.ravel()
    return points, counts, grids


def section_profiles(s: System, solution, xs, n=80):
    """Flux Q(x) and mean pressure on cross-sections x = const (sampled).

    Midpoint rule on an n x n grid restricted to the fluid (phi < c); the
    extension of u_h into the solid part of the cut cells is masked out.
    """
    uh, ph, _ = solution
    flux, pressure = [], []
    for x0 in xs:
        points, _, _ = slice_points(s.params, 0, x0, n)
        fluid = is_fluid(points, s.params.c)
        u_x = core.eval_at_points(uh, points)[:, 0]
        pr = core.eval_at_points(ph, points)
        flux.append(float(np.nansum(np.where(fluid, u_x, 0.0)) / len(points)))
        pressure.append(float(np.nanmean(pr[fluid])))
    return np.array(flux), np.array(pressure)


def velocity_slice(s: System, solution, axis, value, n=80):
    """|u|, u_x and fluid mask on a plane, as 2D arrays (NaN in the solid)."""
    uh, _, _ = solution
    points, counts, grids = slice_points(s.params, axis, value, n)
    fluid = is_fluid(points, s.params.c)
    u = core.eval_at_points(uh, points)
    u[~fluid] = np.nan
    shape = tuple(counts)
    return {"axis": axis, "value": value, "a": grids[0], "b": grids[1],
            "speed": np.linalg.norm(u, axis=1).reshape(shape),
            "u_x": u[:, 0].reshape(shape)}


def quantities(s: System, solution) -> dict:
    """Flux, pressure drop, apparent permeability and divergence."""
    uh, _, _ = solution
    p = s.params
    cells = cell_averages(s, solution)
    # Pressure drop per unit cell between consecutive cells, skipping the
    # first and last cell (inlet and outlet effects) when there are >= 4
    p_cells = cells["pressure"][1:-1] if p.n_x >= 4 else cells["pressure"]
    dp_cell = float(np.mean(-np.diff(p_cells)))
    Q_mean = float(np.mean(cells["flux"]))
    div = core.l2_norm(ufl.div(uh), s.measures.dx)
    return {"Q_in": inlet_flux(p), "Q_mean": Q_mean, "Q_cells": cells["flux"],
            "p_cells": cells["pressure"], "porosity": cells["volume"],
            "dp_cell": dp_cell,
            # Darcy: Q / A = (k / nu) dp / dx with A = 1 and dx = 1 (one cell)
            "k_app": p.nu * Q_mean / dp_cell,
            "div_L2": div}


def save_vtk(path, s: System, solution) -> None:
    """Write velocity and pressure for ParaView / PyVista."""
    uh, ph, _ = solution
    core.save_vtk(path, s.mesh, {"uh": uh, "ph": ph})
