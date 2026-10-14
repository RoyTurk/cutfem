"""Verification of periodic box faces in the Navier-Stokes solver (serial).

Run with ``python tests/test_periodic.py`` (or pytest). Checks:

* the periodic map ties every node on a face x_d = xmax_d to its image on
  x_d = xmin_d (a periodic field is reproduced, a non-periodic one is not),
  also on a gyroid unit cell, where some face nodes have no image;
* optimal convergence orders for a periodic manufactured solution around a
  disk, periodic in x and y (no other condition on the box);
* Poiseuille flow in a pipe, periodic along its axis and driven by a body
  force: exact in Q2 (up to quadrature), flow rate pi G R^4 / (8 nu). The
  cut cells touch the periodic faces, as in a gyroid unit cell.
"""

import sys
from dataclasses import replace

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import qugar.impl
import ufl
from mpi4py import MPI
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

from cutfem import core, study
from cutfem import navier_stokes as ns

PARAMS = ns.Params(nu=0.1, U=1.0)

# Periodic manufactured solution on the unit square (divergence free)


def exact_velocity(x):
    """u = (sin 2 pi x cos 2 pi y, -cos 2 pi x sin 2 pi y)."""
    w = 2 * np.pi
    return ufl.as_vector([ufl.sin(w * x[0]) * ufl.cos(w * x[1]),
                          -ufl.cos(w * x[0]) * ufl.sin(w * x[1])])


def exact_pressure(x):
    """p = sin 2 pi x sin 2 pi y."""
    return ufl.sin(2 * np.pi * x[0]) * ufl.sin(2 * np.pi * x[1])


def mms_system(n, params=PARAMS):
    """Unit square minus a disk, periodic in x and y, Nitsche on the disk."""
    mesh, h = core.disk_mesh(n, (0.0, 0.0), (1.0, 1.0), (0.53, 0.51), 0.25,
                             inside=False)
    x = ufl.SpatialCoordinate(mesh)
    u_ex, p_ex = exact_velocity(x), exact_pressure(x)
    f = (ufl.dot(ufl.grad(u_ex), u_ex) + ufl.grad(p_ex)
         - params.nu * ufl.div(ufl.grad(u_ex)))
    return ns.build(params, mesh, h, f=f, g=u_ex,
                    periodic=((0.0, 0.0), (1.0, 1.0), (0, 1)))


def mms_errors(s, solution):
    """Velocity L2/H1 and mean-free pressure L2 errors (exact data in P6)."""
    uh, ph = solution
    dx = s.measures.dx
    x = ufl.SpatialCoordinate(s.mesh)
    u_ex = ns.data_function(exact_velocity(x), s.mesh, 6, "u_ex")
    p_ex = ns.data_function(exact_pressure(x), s.mesh, 6, "p_ex")
    one = dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0))
    e_p = ph - p_ex
    e_p = e_p - core.integrate(e_p * dx) / core.integrate(one * dx)
    e_u = uh - u_ex
    return {"L2_u": core.l2_norm(e_u, dx), "H1_u": core.h1_seminorm(e_u, dx),
            "L2_p": core.l2_norm(e_p, dx)}


# Poiseuille flow in a pipe along x


def pipe_system(n, params=PARAMS, G=1.0, R=0.35, length=0.5):
    """Pipe of radius R along x in [0, length] x [0, 1]^2, periodic in x.

    Body force (G, 0, 0) as a Function (no Constant next to Functions in a
    cut form), no-slip on the pipe wall by Nitsche.
    """
    axis = np.array([1.0, 0.0, 0.0])
    center = np.array([0.0, 0.52, 0.49])
    domain = qugar.impl.create_cylinder(R, center, axis)
    xmin, xmax = np.zeros(3), np.array([length, 1.0, 1.0])
    mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, domain, [round(length * n), n, n], xmin, xmax,
        exclude_empty_cells=True, ghost_mode=dolfinx.mesh.GhostMode.shared_facet,
        dtype=core.DTYPE)
    W = core.lagrange_space(mesh, 1, shape=(3,))
    f = dolfinx.fem.Function(W, name="f")
    f.interpolate(lambda x: np.vstack([np.full(x.shape[1], G), np.zeros((2, x.shape[1]))]))
    s = ns.build(params, mesh, 1.0 / n, f=f, periodic=(xmin, xmax, (0,)))
    return s, center, R


def pipe_errors(s, solution, center, R, G=1.0):
    """Relative errors of u (L2) and the flow rate, and max |p_h|."""
    uh, ph = solution
    nu, dx = s.params.nu, s.measures.dx
    x = ufl.SpatialCoordinate(s.mesh)
    r2 = (x[1] - center[1]) ** 2 + (x[2] - center[2]) ** 2
    u_ex = ns.data_function(ufl.as_vector([G / (4 * nu) * (R ** 2 - r2), 0.0, 0.0]),
                            s.mesh, 2, "u_ex")
    length = s.mesh.geometry.x[:, 0].max()
    flow_rate = core.integrate(uh[0] * dx) / length      # mean over cross-sections
    exact_rate = np.pi * G * R ** 4 / (8 * nu)
    return {"L2_u": core.l2_norm(uh - u_ex, dx) / core.l2_norm(u_ex, dx),
            "flow_rate": abs(flow_rate - exact_rate) / exact_rate,
            "max_p": float(np.max(np.abs(ph.x.array)))}


# Tests


def group_sizes(P):
    """Number of full dofs per periodic unknown (column sums of P)."""
    ones, sizes = P.createVecLeft(), P.createVecRight()
    ones.set(1.0)
    P.multTranspose(ones, sizes)
    return sizes


def restriction_defect(V, Q, P, expr_u, expr_p):
    """max |P R x - x| for x = (expr_u, expr_p) interpolated, R averaging a group."""
    u, p = dolfinx.fem.Function(V), dolfinx.fem.Function(Q)
    u.interpolate(expr_u)
    p.interpolate(expr_p)
    full = P.createVecLeft()
    n_u = len(u.x.array)
    full.array[:n_u], full.array[n_u:] = u.x.array, p.x.array
    reduced, back = P.createVecRight(), P.createVecLeft()
    P.multTranspose(full, reduced)
    reduced.pointwiseDivide(reduced, group_sizes(P))
    P.mult(reduced, back)
    return np.max(np.abs(back.array - full.array))


def test_periodic_map():
    """A periodic field equals its restriction-prolongation; x itself does not."""
    with mms_system(16) as s:
        n_full, n_red = s.P.getSize()
        print(f"  {n_full} dofs, {n_red} after identification")
        w = 2 * np.pi
        periodic = restriction_defect(
            s.V, s.Q, s.P, lambda x: np.vstack([np.sin(w * x[0]), np.cos(w * x[1])]),
            lambda x: np.cos(w * x[0]) * np.sin(w * x[1]))
        linear = restriction_defect(s.V, s.Q, s.P, lambda x: np.vstack([x[0], x[1]]),
                                    lambda x: x[0] + x[1])
        print(f"  periodic field: defect {periodic:.1e}; x, y: defect {linear:.1e}")
        assert periodic < 1e-12
        assert linear > 0.5


def test_periodic_map_gyroid():
    """Gyroid unit cell: nodes without image (empty cell across the face) stay free.

    Excluded empty cells make the active mesh differ on opposite faces;
    groups of 3 or 7 nodes (instead of 2, 4, 8) show the incomplete ones.
    """
    lo, hi = np.zeros(3), np.ones(3)
    mesh = create_unfitted_impl_Cartesian_mesh(
        MPI.COMM_WORLD, qugar.impl.create_Schoen(periods=[1.0, 1.0, 1.0]), [8] * 3, lo, hi,
        exclude_empty_cells=True, dtype=core.DTYPE)
    V, Q = core.lagrange_space(mesh, 2, (3,)), core.lagrange_space(mesh, 1)
    P = ns.periodic_map(V, Q, lo, hi, (0, 1, 2))
    sizes, counts = np.unique(group_sizes(P).array.astype(int), return_counts=True)
    w = 2 * np.pi
    defect = restriction_defect(
        V, Q, P, lambda x: np.vstack([np.sin(w * x[0]), np.cos(w * x[1]),
                                      np.sin(w * x[2] + 1)]),
        lambda x: np.cos(w * x[0]) * np.sin(w * x[1]) * np.cos(w * x[2]))
    print(f"  {P.getSize()[0]} -> {P.getSize()[1]} dofs, group sizes "
          f"{dict(zip(sizes.tolist(), counts.tolist(), strict=True))}, defect {defect:.1e}")
    assert defect < 1e-12
    assert {3, 7} & set(sizes.tolist())
    P.destroy()


def test_mms_periodic():
    """Optimal orders for the periodic manufactured solution."""
    rows = []
    for n in (16, 32, 64):
        with mms_system(n) as s:
            solution = ns.solve(s)
            rows.append({"h": s.h, **mms_errors(s, solution)})
            print(f"  n = {n:3d}: newton {len(s.history) - 1}, "
                  + ", ".join(f"{k} {v:.3e}" for k, v in rows[-1].items() if k != "h"))
    h = np.array([r["h"] for r in rows])
    for key, expected in (("L2_u", 3.0), ("H1_u", 2.0), ("L2_p", 2.0)):
        order = study.rates(h, [r[key] for r in rows])
        print(f"    {key}: orders {np.round(order, 2)} (expected {expected})")
        assert order[-1] > expected - 0.3, (key, order)


def test_poiseuille_pipe():
    """Periodic pipe flow: Q2 reproduces the paraboloid, exact flow rate.

    Quadrature degree 10: the default (5) does not integrate the Q2 x Q2
    products on cut cells exactly, which alone leaves an error ~4e-5 (n = 8).
    """
    params = replace(PARAMS, quadrature_degree=10)
    for n in (8, 12):
        s, center, R = pipe_system(n, params)
        with s:
            solution = ns.solve(s)
            errors = pipe_errors(s, solution, center, R)
            print(f"  n = {n}: {s.n_dofs} dofs, newton {len(s.history) - 1}, "
                  + ", ".join(f"{k} {v:.2e}" for k, v in errors.items()))
        assert errors["L2_u"] < 1e-7
        assert errors["flow_rate"] < 1e-9
        assert errors["max_p"] < 1e-10


if __name__ == "__main__":
    tests = [(name, f) for name, f in globals().items() if name.startswith("test_")]
    failed = []
    for name, test in tests:
        print(name)
        try:
            test()
            print("  ok")
        except Exception as error:                       # noqa: BLE001
            failed.append(name)
            print(f"  FAILED: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
