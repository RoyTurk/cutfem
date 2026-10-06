"""PDE problems.

Every problem module follows the same interface, so the drivers in
:mod:`cutfem.study` work for all of them:

``Params``
    Frozen dataclass with every parameter of a run (mesh, geometry,
    stabilization, discretization). Vary one with ``dataclasses.replace``.
``build(params) -> System``
    Assemble the system. Use as ``with build(params) as system:`` so the
    PETSc objects are freed.
``solve(system) -> solution``
    Solve; must not modify ``system.A``.
``errors(system, solution) -> dict[str, float]``
    Error norms against the exact solution.
``ERRORS``
    Tuple of the keys returned by ``errors`` (NaN-filled if a solve fails).
``expected_rates(params) -> dict[str, float]``
    Theoretical convergence orders of the keys returned by ``errors``.
"""
