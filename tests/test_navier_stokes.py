"""Verification of the Navier-Stokes CutFEM machinery (cutfem.navier_stokes).

Run with ``python tests/test_navier_stokes.py`` (or pytest). Checks:

* the assembled Jacobian is the derivative of the assembled residual
  (central finite differences, strong BCs and ghost penalty included);
* Newton converges quadratically, Picard + Newton reaches the same solution;
* the force (Nitsche flux on Gamma) equals the variational volume force;
* results do not depend on PYTHONHASHSEED (qugar quadrature ordering);
* optimal convergence orders for a manufactured solution, internal only,
  with Nitsche everywhere (pressure pinned) and with strong Dirichlet +
  Neumann + Nitsche (channel with a hole).
"""

import os
import subprocess
import sys
from dataclasses import replace

import dolfinx.fem
import dolfinx.mesh
import numpy as np
import ufl

from cutfem import core, study
from cutfem import navier_stokes as ns
from cutfem.problems import dfg2d1

# Manufactured solution (divergence free): u = (20 x y^3, 5 x^4 - 5 y^4),
# p = 60 x^2 y - 20 y^3; f = (u . grad) u + grad p - nu Delta u


def exact_velocity(x):
    """Exact velocity for UFL coordinates or NumPy arrays (shape (3, n))."""
    components = [20 * x[0] * x[1] ** 3, 5 * x[0] ** 4 - 5 * x[1] ** 4]
    if isinstance(x, np.ndarray):
        return np.vstack(components)
    return ufl.as_vector(components)


def exact_pressure(x):
    """Exact pressure."""
    return 60 * x[0] ** 2 * x[1] - 20 * x[1] ** 3


def mms_system(geometry, n, params):
    """Navier-Stokes system of the manufactured solution.

    ``"disk"``: disk of radius 0.5 in [-1, 1]^2, Nitsche on the whole
    boundary, pressure pinned. ``"channel"``: [-0.5, 0.5]^2 minus a disk of
    radius 0.2, strong Dirichlet at x = -0.5 and y = +-0.5, Neumann at
    x = 0.5, Nitsche on the hole.
    """
    center = (0.03, 0.01)
    if geometry == "disk":
        mesh, h = core.disk_mesh(n, (-1.0, -1.0), (1.0, 1.0), center, 0.5)
    else:
        mesh, h = core.disk_mesh(n, (-0.5, -0.5), (0.5, 0.5), center, 0.2,
                                 inside=False)
    x = ufl.SpatialCoordinate(mesh)
    u_ex, p_ex = exact_velocity(x), exact_pressure(x)
    f = (ufl.dot(ufl.grad(u_ex), u_ex) + ufl.grad(p_ex)
         - params.nu * ufl.div(ufl.grad(u_ex)))
    if geometry == "disk":
        return ns.build(params, mesh, h, f=f, g=u_ex)

    walls = ns.boundary_facets(
        mesh, lambda y: np.isclose(y[0], -0.5) | np.isclose(np.abs(y[1]), 0.5))
    outlet = ns.boundary_facets(mesh, lambda y: np.isclose(y[0], 0.5))
    tags = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, outlet,
                                 np.ones(len(outlet), dtype=np.int32))
    ds_out = ufl.Measure("ds", domain=mesh, subdomain_data=tags, subdomain_id=1)
    n_F = ufl.FacetNormal(mesh)
    g_N = params.nu * ufl.dot(ufl.grad(u_ex), n_F) - p_ex * n_F
    return ns.build(params, mesh, h, f=f, g=u_ex,
                    strong=[(walls, exact_velocity)], neumann=[(ds_out, g_N)])


def mms_errors(s, solution):
    """Velocity L2/H1 errors and pressure L2 error (mean-free if pinned).

    The exact solution is interpolated into P6 (exact for these
    polynomials): cut integrals mixing the P1 pressure with expressions of
    x are wrong in qugar (see ``ns.data_function``).
    """
    uh, ph = solution
    dx = s.measures.dx
    x = ufl.SpatialCoordinate(s.mesh)
    u_ex = ns.data_function(exact_velocity(x), s.mesh, 6, "u_ex")
    p_ex = ns.data_function(exact_pressure(x), s.mesh, 6, "p_ex")
    e_p = ph - p_ex
    if s.pinned:
        one = dolfinx.fem.Constant(s.mesh, core.DTYPE(1.0))
        e_p = e_p - core.integrate(e_p * dx) / core.integrate(one * dx)
    e_u = uh - u_ex
    return {"L2_u": core.l2_norm(e_u, dx), "H1_u": core.h1_seminorm(e_u, dx),
            "L2_p": core.l2_norm(e_p, dx)}


# Tests


def _jacobian_error(s):
    """|J d - (R(x + eps d) - R(x - eps d)) / (2 eps)| / |.| on the free dofs."""
    rng = np.random.default_rng(0)
    s.uh.x.array[:] = 0.1 * rng.standard_normal(s.uh.x.array.size)
    s.ph.x.array[:] = 0.1 * rng.standard_normal(s.ph.x.array.size)
    n_u = s.V.dofmap.index_map.size_local * s.V.dofmap.index_map_bs
    velocity_bcs = s.bcs[:-1] if s.pinned else s.bcs
    for bc in velocity_bcs:                    # the state satisfies the strong BCs
        bc.set(s.uh.x.array)
    constrained = [bc.dof_indices()[0] for bc in velocity_bcs]
    if s.pinned:
        constrained.append(n_u + s.bcs[-1].dof_indices()[0])
    constrained = np.concatenate([np.zeros(0, dtype=np.int32), *constrained])

    ns.assemble_jacobian(s)
    d = rng.standard_normal(s.A.getSize()[0])
    d[constrained] = 0.0
    x, y = s.A.createVecRight(), s.A.createVecLeft()
    x.array[:] = d
    s.A.mult(x, y)

    def residual(scale):
        s.uh.x.array[:n_u] += scale * d[:n_u]
        s.ph.x.array[:d.size - n_u] += scale * d[n_u:]
        ns.assemble_residual(s)
        r = s.b.array.copy()
        s.uh.x.array[:n_u] -= scale * d[:n_u]
        s.ph.x.array[:d.size - n_u] -= scale * d[n_u:]
        return r

    eps = 1e-5
    fd = (residual(eps) - residual(-eps)) / (2 * eps)
    free = np.setdiff1d(np.arange(d.size), constrained)
    return np.linalg.norm(y.array[free] - fd[free]) / np.linalg.norm(fd[free])


def test_jacobian_is_derivative_of_residual():
    """Jacobian = derivative of the residual: DFG, and with x-dependent data."""
    params = ns.Params(nu=0.1, U=1.0)
    systems = {"dfg2d1": lambda: dfg2d1.build(dfg2d1.Params(n_y=16)),
               "mms channel": lambda: mms_system("channel", 8, params),
               "mms disk (pinned)": lambda: mms_system("disk", 16, params)}
    for name, build in systems.items():
        with build() as s:
            error = _jacobian_error(s)
        print(f"  {name}: relative Jacobian error {error:.2e}")
        assert error < 1e-8, name


def test_newton_converges_quadratically():
    """Residual exponents roughly double; Picard + Newton gives the same flow."""
    with dfg2d1.build(dfg2d1.Params(n_y=24)) as s:
        u, p = (f.x.array.copy() for f in dfg2d1.solve(s))
        history = np.array(s.history)
        print("  Newton:", " ".join(f"{r:.1e}" for r in history))
        assert len(history) - 1 <= 8
        # order estimates log(r_{k+1} / r_k) / log(r_k / r_{k-1}) from the
        # Stokes iterate on (step 0 removes the strong BC mismatch)
        reduction = np.log(history[2:] / history[1:-1])
        orders = reduction[1:] / reduction[:-1]
        print("  order estimates:", np.round(orders, 2))
        assert orders[-1] > 1.8, orders

    with dfg2d1.build(dfg2d1.Params(n_y=24, n_picard=3)) as s:
        u2, p2 = (f.x.array.copy() for f in dfg2d1.solve(s))
        print("  Picard + Newton:", " ".join(f"{r:.1e}" for r in s.history))
        assert np.linalg.norm(u2 - u) <= 1e-9 * np.linalg.norm(u)
        assert np.linalg.norm(p2 - p) <= 1e-9 * np.linalg.norm(p)


def test_force_equals_volume_force():
    """Nitsche flux on Gamma = variational force with v_e = e near Gamma."""
    with dfg2d1.build(dfg2d1.Params(n_y=32)) as s:
        solution = dfg2d1.solve(s)
        F = ns.force(s, solution)
        for i, e in enumerate(np.eye(2)):
            v_e = ns.force_test_function(s, e)
            F_vol = ns.volume_force(s, solution, v_e)
            print(f"  F[{i}]: boundary {F[i]:.12e}, volume {F_vol:.12e}")
            assert abs(F[i] - F_vol) <= 1e-8 * abs(F[0])


def test_condition_number_matches_dense_svd():
    """SLEPc kappa on the free dofs = dense SVD (strong BCs and pressure pin)."""
    params = ns.Params(nu=0.1, U=1.0)
    systems = {"dfg2d1": lambda: dfg2d1.build(dfg2d1.Params(n_y=8)),
               "mms disk (pinned)": lambda: mms_system("disk", 8, params)}
    for name, build in systems.items():
        with build() as s:
            ns.solve(s)
            kappa, _, _ = ns.condition_number(s)
            free = ns.free_dofs(s)
            sigma = np.linalg.svd(core.to_scipy(s.A).toarray()[np.ix_(free, free)],
                                  compute_uv=False)
            print(f"  {name}: kappa {kappa:.6e} (SLEPc), {sigma[0] / sigma[-1]:.6e} (dense)")
            assert abs(kappa - sigma[0] / sigma[-1]) <= 1e-6 * kappa, name


def test_independent_of_hash_seed():
    """Same drag and lift for different PYTHONHASHSEED (qugar rule order)."""
    script = ("from cutfem.problems import dfg2d1\n"
              "with dfg2d1.build(dfg2d1.Params(n_y=16)) as s:\n"
              "    b = dfg2d1.benchmark(s, dfg2d1.solve(s))\n"
              "print(repr((b['c_D'], b['c_L'])))\n")
    values = []
    for seed in ("0", "1", "2"):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, check=True,
                             env={**os.environ, "PYTHONHASHSEED": seed})
        values.append(np.array(eval(out.stdout.strip().splitlines()[-1],
                                    {"np": np})))
        print(f"  seed {seed}: c_D, c_L = {values[-1]}")
    for v in values[1:]:
        assert np.allclose(v, values[0], rtol=1e-10, atol=0.0)


def _mms_convergence(geometry, n_cells, params):
    rows = []
    for n in n_cells:
        with mms_system(geometry, n, params) as s:
            solution = ns.solve(s)
            rows.append({"h": s.h, **mms_errors(s, solution)})
            print(f"  {geometry} n = {n:3d}: newton {len(s.history) - 1}, "
                  + ", ".join(f"{k} {v:.3e}" for k, v in rows[-1].items() if k != "h"))
    h = np.array([r["h"] for r in rows])
    observed = {}
    for key, expected in (("L2_u", 3.0), ("H1_u", 2.0), ("L2_p", 2.0)):
        order = study.rates(h, [r[key] for r in rows])
        observed[key] = order
        print(f"    {key}: orders {np.round(order, 2)} (expected {expected})")
        assert order[-1] > expected - 0.3, (key, order)
    return observed


def test_mms_nitsche_only():
    """Optimal orders, Nitsche on the whole boundary (pressure pinned)."""
    params = ns.Params(nu=0.1, U=1.0)
    _mms_convergence("disk", [16, 32, 64], params)


def test_mms_strong_neumann_nitsche():
    """Optimal orders with strong Dirichlet, Neumann and Nitsche boundaries."""
    params = ns.Params(nu=0.1, U=1.0)
    _mms_convergence("channel", [8, 16, 32], params)


def test_mms_normal_penalty():
    """The optional Nitsche normal penalty keeps the orders optimal."""
    params = replace(ns.Params(nu=0.1, U=1.0), gamma_n=10.0)
    _mms_convergence("channel", [8, 16, 32], params)


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
