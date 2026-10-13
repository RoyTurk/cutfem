"""Stationary flow around a cylinder, DFG benchmark 2D-1 (Re = 20).

    (u . grad) u + grad p - nu Delta u = 0,   div u = 0   in Omega,

Omega = [0, 2.2] x [0, 0.41] minus the disk of radius 0.05 at (0.2, 0.2),
nu = 1e-3, rho = 1. Parabolic inflow u = (4 U_m y (H - y) / H^2, 0) at x = 0
and no-slip on the walls y = 0, H, both imposed strongly; do-nothing outflow
nu du/dn - p n = 0 at x = 2.2; no-slip on the cylinder by Nitsche.

Reference values (Nabh, FEATFLOW benchmark page):
c_D = 5.57953523384, c_L = 0.010618948146.
"""

from dataclasses import dataclass

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import ufl

from cutfem import core
from cutfem import navier_stokes as ns

REFERENCE = {"c_D": 5.57953523384, "c_L": 0.010618948146}
QUANTITIES = tuple(REFERENCE)
ERRORS = tuple(f"err_{key}" for key in QUANTITIES)
OUTLET = 1                      # facet tag of the outflow boundary


@dataclass(frozen=True)
class Params(ns.Params):
    """Parameters of one DFG 2D-1 run (adds geometry and inflow to ns.Params)."""

    nu: float = 1e-3
    U: float = 0.2              # mean inflow velocity, ghost penalty weight
    U_m: float = 0.3            # maximum inflow velocity
    n_y: int = 32               # cells across the channel
    n_x: int | None = None      # cells along the channel; None: square cells
    center: tuple = (0.2, 0.2)
    radius: float = 0.05
    length: float = 2.2
    height: float = 0.41


def channel_mesh(params: Params):
    """Cartesian background mesh of the channel, n_y cells across its height.

    By default n_x = round(L n_y / H), so the cells are square up to the
    rounding of n_x; ``params.n_x`` overrides it (slightly non-square cells,
    a different cut of the cylinder at the same geometry). Returns ``(mesh, h)``
    with h the largest cell side.
    """
    n_x = params.n_x or round(params.length * params.n_y / params.height)
    return core.disk_mesh([n_x, params.n_y], (0.0, 0.0),
                          (params.length, params.height),
                          params.center, params.radius, inside=False)


def build(params: Params) -> ns.System:
    """Assemble the DFG 2D-1 Navier-Stokes system."""
    mesh, h = channel_mesh(params)
    L, H = params.length, params.height

    def inflow(x):
        return np.vstack((4.0 * params.U_m * x[1] * (H - x[1]) / H ** 2,
                          np.zeros_like(x[0])))

    # Inlet and walls in one BC: the inflow profile vanishes at y = 0, H
    walls = ns.boundary_facets(
        mesh, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0)
        | np.isclose(x[1], H))
    outlet = ns.boundary_facets(mesh, lambda x: np.isclose(x[0], L))
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, outlet,
                                 np.full(len(outlet), OUTLET, dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=mesh, subdomain_data=tags,
                         subdomain_id=OUTLET)
    do_nothing = dolfinx.fem.Constant(mesh, np.zeros(2, dtype=core.DTYPE))

    return ns.build(params, mesh, h, strong=[(walls, inflow)],
                    neumann=[(ds_out, do_nothing)])


def solve(s: ns.System, verbose=False):
    """Newton from zero (first step: Stokes); returns ``(uh, ph)``."""
    return ns.solve(s, verbose=verbose)


def benchmark(s: ns.System, solution) -> dict:
    """Drag and lift coefficients.

    c_D, c_L = 2 F / (U_mean^2 D) with U_mean = 2/3 U_m, D = 2 r and F the
    Nitsche-consistent force on the cylinder (:func:`ns.force`).
    """
    params = s.params
    F = ns.force(s, solution)
    scale = 2.0 / ((2.0 / 3.0 * params.U_m) ** 2 * 2.0 * params.radius)
    return {"c_D": scale * F[0], "c_L": scale * F[1]}


def errors(s: ns.System, solution) -> dict:
    """Relative errors of the benchmark quantities against ``REFERENCE``."""
    values = benchmark(s, solution)
    return {f"err_{key}": abs(values[key] - ref) / abs(ref)
            for key, ref in REFERENCE.items()}


def expected_rates(params: Params) -> dict:
    """No proven orders for the benchmark quantities."""
    return {}


def save_vtk(path, s: ns.System, solution) -> None:
    """Write velocity and pressure for ParaView."""
    uh, ph = solution
    core.save_vtk(path, s.mesh, {"uh": uh, "ph": ph})
