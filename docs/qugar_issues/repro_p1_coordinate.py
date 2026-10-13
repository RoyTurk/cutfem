"""qugar issue 2: wrong cut integrals of a P1 Function times the y-coordinate.

On a quadrilateral mesh, the cut-cell integral of f * g(x), with f a P1
(Q1) Function and g an expression of the spatial coordinates containing
x[1], is wrong. The same integral with f copied exactly into P6 is correct.
The result does not depend on PYTHONHASHSEED. Run:

    python repro_p1_coordinate.py
"""

import numpy as np
import ufl
from mpi4py import MPI

import basix.ufl
import dolfinx.fem
import qugar.dolfinx
import qugar.impl
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

disk = qugar.impl.create_disk(0.5, center=np.array([0.03, 0.01]))
mesh = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, disk, [16, 16], np.array([-1.0, -1.0]), np.array([1.0, 1.0]),
    exclude_empty_cells=True, dtype=np.float64)


def space(degree):
    return dolfinx.fem.functionspace(
        mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), degree))


def integrate(expr):
    form = qugar.dolfinx.form_custom(expr * ufl.dx(domain=mesh))
    local = dolfinx.fem.assemble_scalar(form, coeffs=form.pack_coefficients())
    return mesh.comm.allreduce(local, op=MPI.SUM)


x = ufl.SpatialCoordinate(mesh)
for degree in (1, 2):
    f = dolfinx.fem.Function(space(degree))
    f.interpolate(lambda y: np.sin(2 * y[0]) + y[1])
    f6 = dolfinx.fem.Function(space(6))                 # exact copy of f in P6
    f6.interpolate(f)
    print(f"f in P{degree}:")
    for name, g in (("x0", x[0]), ("x0^2", x[0] ** 2), ("x0 x1", x[0] * x[1]),
                    ("x1", x[1]), ("x1^2", x[1] ** 2), ("x0^2 x1", x[0] ** 2 * x[1]),
                    ("sin(x1)", ufl.sin(x[1]))):
        a, b = integrate(f * g), integrate(f6 * g)
        print(f"  int f * {name:8s}: {a: .10f}   with f in P6: {b: .10f}   "
              f"rel. difference {abs(a - b) / abs(b):.1e}")
