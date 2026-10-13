"""Pressure-driven flow through a Schoen gyroid channel (3D feasibility study).

    (u . grad) u + grad p - nu Delta u = 0,   div u = 0   in Omega,

Omega = {phi <= 0} inside the box [0, N L] x [0, L] x [0, L], with phi the
Schoen gyroid of unit-cell size L (one of the two labyrinths, porosity 1/2),
N unit cells in the flow direction x. Boundary conditions:

* x = 0 (inlet):    pseudo-traction nu du/dn - p n = -p_in n;
* x = N L (outlet): do-nothing, nu du/dn - p n = 0;
* y = 0, L and z = 0, L: free-slip, zero normal velocity component (strong);
* gyroid surface Gamma: no-slip by Nitsche.

The inlet and outlet faces are cut by the gyroid, so their traction terms use
cut quadrature on the exterior facets of the background mesh.

QUGaR's ``create_Schoen(periods)`` evaluates
    phi = sin(a x) cos(b y) + sin(b y) cos(c z) + sin(c z) cos(a x),
    (a, b, c) = 2 pi periods,
in physical coordinates, so ``periods = [1/L] * 3`` gives unit cells of size
L whatever the box (checked against the surface area: N x 1 x 1 cells give
exactly N times the unit-cell area).
"""

import time
from dataclasses import dataclass

import basix.ufl
import dolfinx.fem
import dolfinx.mesh
import numpy as np
import qugar.impl
import ufl
from mpi4py import MPI
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from cutfem import core
from cutfem import navier_stokes as ns

INLET, OUTLET = 1, 2                # facet tags
DIAGNOSTIC_DEGREE = 8               # cut quadrature of the reported integrals
QUANTITIES = ("Q_in", "Q_out", "mass_balance", "wall_flux", "divergence", "U_s",
              "permeability", "Re")
ERRORS = ()


@dataclass(frozen=True)
class Params(ns.Params):
    """Parameters of one gyroid run (adds geometry and driving pressure)."""

    nu: float = 0.01
    U: float = 0.1              # characteristic velocity (ghost penalty weight)
    gamma_u: float = 0.01       # ghost penalties from the DFG 2D-1 study
    gamma_p: float = 0.01
    p_in: float = 1.0           # inlet pressure (outlet: 0)
    L: float = 1.0              # unit-cell size
    N: int = 3                  # unit cells in the flow direction
    n_cell: int = 8             # background cells per unit-cell length


def gyroid_mesh(params: Params):
    """Unfitted Cartesian mesh of the box cut by the gyroid.

    Returns ``(mesh, h, seconds)``: h the cell size and the wall time of the
    mesh creation (analytic, non-Bezier quadrature path for TPMS).
    """
    L, N, n = params.L, params.N, params.n_cell
    domain = qugar.impl.create_Schoen(periods=[1.0 / L] * 3)
    start = time.perf_counter()
    mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, domain, [N * n, n, n], np.zeros(3),
        np.array([N * L, L, L]), exclude_empty_cells=True,
        ghost_mode=dolfinx.mesh.GhostMode.shared_facet, dtype=core.DTYPE)
    return mesh, L / n, time.perf_counter() - start


def build(params: Params) -> ns.System:
    """Assemble the gyroid Navier-Stokes system."""
    mesh, h, _ = gyroid_mesh(params)
    L, length = params.L, params.N * params.L
    fdim = mesh.topology.dim - 1

    inlet = ns.boundary_facets(mesh, lambda x: np.isclose(x[0], 0.0))
    outlet = ns.boundary_facets(mesh, lambda x: np.isclose(x[0], length))
    facets = np.concatenate([inlet, outlet])
    values = np.concatenate([np.full(len(inlet), INLET, dtype=np.int32),
                             np.full(len(outlet), OUTLET, dtype=np.int32)])
    order = np.argsort(facets)
    tags = dolfinx.mesh.meshtags(mesh, fdim, facets[order], values[order])
    # Cut faces need the fixed cut quadrature degree too: the automatic one
    # (2 for a Q2 test function) misses ~0.2 % of the inlet traction
    degree = params.quadrature_degree or 3 * params.v_degree - 1
    measure = dict(domain=mesh, subdomain_data=tags,
                   metadata={"quadrature_degree": degree})
    ds_in = ufl.Measure("ds", subdomain_id=INLET, **measure)
    ds_out = ufl.Measure("ds", subdomain_id=OUTLET, **measure)

    # nu du/dn - p n = -p_in n with n = (-1, 0, 0) at x = 0: g_N = (p_in, 0, 0)
    g_in = dolfinx.fem.Constant(mesh, np.array([params.p_in, 0.0, 0.0],
                                               dtype=core.DTYPE))
    slip = [(ns.boundary_facets(mesh, lambda x, i=i, c=c: np.isclose(x[i], c)), i)
            for i in (1, 2) for c in (0.0, L)]
    s = ns.build(params, mesh, h, slip=slip,
                 neumann=[(ds_in, g_in), (ds_out, None)], neumann_cut=True)
    s.measures_bc = {"inlet": ds_in, "outlet": ds_out}
    return s


def solve(s: ns.System, verbose=False):
    """Newton from zero (first step: Stokes); returns ``(uh, ph)``."""
    return ns.solve(s, verbose=verbose)


def benchmark(s: ns.System, solution) -> dict:
    """Flow rates, mass balance, superficial velocity, permeability, Reynolds.

    Q_in = int_inlet u_x, Q_out = int_outlet u_x (cut facet integrals over the
    fluid part of the faces). By the divergence theorem
    Q_out - Q_in + int_Gamma u.n = int_Omega div u (free-slip faces carry no
    flux), so the imbalance splits into the flux through the weakly imposed
    no-slip wall and the discrete divergence, both reported relative to Q.
    These integrals use cut quadrature of degree DIAGNOSTIC_DEGREE: with the
    solver's degree, int div u = int_Gamma u.n holds exactly (continuity
    tested with q = 1) and hides the imbalance.
    U_s = Q_out / L^2 (superficial velocity, box
    cross-section); apparent Darcy permeability k = nu U_s (N L) / p_in
    (rho = 1, outlet pressure 0); Re = U_s L / nu.
    """
    uh, _ = solution
    params = s.params
    md = {"quadrature_degree": DIAGNOSTIC_DEGREE}
    m = core.cut_measures(s.mesh)
    Q_in = core.integrate(uh[0] * s.measures_bc["inlet"](metadata=md))
    Q_out = core.integrate(uh[0] * s.measures_bc["outlet"](metadata=md))
    wall = core.integrate(ufl.dot(uh, m.n) * m.ds(metadata=md))     # weak no-slip
    divergence = core.integrate(ufl.div(uh) * m.dx(metadata=md))
    U_s = Q_out / params.L ** 2
    return {"Q_in": Q_in, "Q_out": Q_out,
            "mass_balance": abs(Q_in - Q_out) / abs(Q_out),
            "wall_flux": wall / abs(Q_out),          # int_Gamma u.n, relative to Q
            "divergence": divergence / abs(Q_out),   # int_Omega div u, relative to Q
            "U_s": U_s,
            "permeability": params.nu * U_s * params.N * params.L / params.p_in,
            "Re": U_s * params.L / params.nu}


def layer_profiles(s: ns.System, solution) -> dict:
    """Flow rate, mean pressure and fluid fraction of each cell layer along x.

    For the slab x_k < x < x_k + h between two grid planes, int u_x dV / h is
    the mean flow rate through its cross-sections, int p dV / |slab cap Omega|
    the mean pressure, and |slab cap Omega| / (h L^2) the mean fluid fraction.
    All three are exact cut integrals of the finite-element solution,
    assembled at once with a DG0 test function and summed per layer (cut
    integrals on interior facets, i.e. on the grid planes themselves, fail in
    QUGaR on 3D hexahedral meshes).
    """
    uh, ph = solution
    mesh, h, params = s.mesh, s.h, s.params
    W = dolfinx.fem.functionspace(mesh, basix.ufl.element("DG", mesh.basix_cell(), 0))
    w = ufl.TestFunction(W)
    dx = core.cut_measures(mesh).dx(metadata={"quadrature_degree": DIAGNOSTIC_DEGREE})
    one = dolfinx.fem.Constant(mesh, core.DTYPE(1.0))
    n_owned = mesh.topology.index_map(3).size_local
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 3, np.arange(n_owned, dtype=np.int32))
    layer = np.floor(midpoints[:, 0] / h).astype(int)
    n_layers = params.N * params.n_cell
    totals = []
    for integrand in (uh[0] * w, ph * w, one * w):
        vector = core.assemble_vector(integrand * dx)
        per_layer = np.bincount(layer, weights=vector.array[:n_owned], minlength=n_layers)
        totals.append(mesh.comm.allreduce(per_layer, op=MPI.SUM))
        vector.destroy()
    flux, pressure, volume = totals
    return {"x": (np.arange(n_layers) + 0.5) * h, "Q": flux / h,
            "p_mean": pressure / volume, "fraction": volume / (h * params.L ** 2)}


def errors(s: ns.System, solution) -> dict:
    """No reference solution."""
    return {}


def expected_rates(params: Params) -> dict:
    """No reference solution."""
    return {}


def geometry(params: Params) -> dict:
    """Step 0 sanity check of the cut geometry.

    Fluid volume (exactly half the box by symmetry), immersed surface area
    per unit-cell volume, smallest cut-cell volume fraction |T cap Omega|/|T|,
    numbers of active and cut cells, and mesh / quadrature creation times.
    """
    mesh, h, t_mesh = gyroid_mesh(params)
    m = core.cut_measures(mesh)
    metadata = {"quadrature_degree": params.quadrature_degree or 3 * params.v_degree - 1}
    dx, ds = m.dx(metadata=metadata), m.ds(metadata=metadata)
    one = dolfinx.fem.Constant(mesh, core.DTYPE(1.0))
    box = params.N * params.L ** 3
    start = time.perf_counter()
    volume = core.integrate(one * dx)
    area = core.integrate(one * ds)
    inlet = ns.boundary_facets(mesh, lambda x: np.isclose(x[0], 0.0))
    tags = dolfinx.mesh.meshtags(mesh, 2, inlet, np.full(len(inlet), INLET, dtype=np.int32))
    ds_in = ufl.Measure("ds", domain=mesh, subdomain_data=tags, subdomain_id=INLET)
    x = ufl.SpatialCoordinate(mesh)
    inlet_area = core.integrate(one * ds_in)
    inlet_moment = core.integrate(x[1] ** 2 * x[2] * ds_in)   # not fixed by symmetry
    W = dolfinx.fem.functionspace(mesh, basix.ufl.element("DG", mesh.basix_cell(), 0))
    volumes = core.assemble_vector(ufl.TestFunction(W) * dx)
    cut = mesh.get_cut_cells()
    local = float(np.min(volumes.array[cut])) / h ** 3 if len(cut) else 1.0
    volumes.destroy()
    t_quad = time.perf_counter() - start
    comm = mesh.comm
    n_cells = mesh.topology.index_map(3).size_local
    return {"n_cell": params.n_cell, "h": h,
            "active_cells": comm.allreduce(n_cells, op=MPI.SUM),
            "cut_cells": comm.allreduce(len(cut), op=MPI.SUM),
            "volume": volume, "volume_error": abs(volume - 0.5 * box) / (0.5 * box),
            "area_per_cell_volume": area / box * params.L,
            "inlet_area": inlet_area, "inlet_moment": inlet_moment,
            "min_cut_fraction": comm.allreduce(local, op=MPI.MIN),
            "t_mesh": t_mesh, "t_quadrature": t_quad}


def save_vtk(path, s: ns.System, solution, degree=2) -> None:
    """Write velocity and pressure for ParaView (fitted reparametrization)."""
    uh, ph = solution
    core.save_vtk(path, s.mesh, {"uh": uh, "ph": ph}, degree=degree)
