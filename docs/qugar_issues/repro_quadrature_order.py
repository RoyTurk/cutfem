"""qugar issue 1: wrong or NaN values for forms mixing dx and dsu integrals.

A form that contains (i) a cut-cell integral (dx) and an unfitted-boundary
integral (dsu), (ii) a Function coefficient and (iii) a Constant can assemble
to wrong values or NaN, depending on PYTHONHASHSEED. Run:

    for s in 0 1 2 3; do PYTHONHASHSEED=$s python repro_quadrature_order.py; done

Each line compares the form with the same form in which the Constant is
replaced by a Function with the same value (always correct, see the last
column): the relative difference should be ~1e-16.
"""

import numpy as np
import ufl
from mpi4py import MPI

import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import qugar.dolfinx
import qugar.impl
from qugar.dolfinx import UnfittedNormal, dsu
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

# Unit square minus a disk
disk = qugar.impl.create_disk(0.2, center=np.array([0.53, 0.51]))
mesh = create_unfitted_impl_Cartesian_mesh(
    MPI.COMM_WORLD, qugar.impl.create_negative(disk), [16, 16],
    np.array([0.0, 0.0]), np.array([1.0, 1.0]), exclude_empty_cells=True,
    dtype=np.float64)

V = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,)))
Q = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 1))
q = ufl.TestFunction(Q)
uh = dolfinx.fem.Function(V)
uh.x.array[:] = np.random.default_rng(0).standard_normal(uh.x.array.size)

c = dolfinx.fem.Constant(mesh, np.array([0.3, -0.2]))      # a Constant ...
c_fun = dolfinx.fem.Function(V)                              # ... and an equal Function
c_fun.interpolate(lambda x: np.vstack((np.full_like(x[0], 0.3), np.full_like(x[0], -0.2))))

n = UnfittedNormal(mesh)
dx = ufl.dx(domain=mesh, metadata={"quadrature_degree": 5})
ds = dsu(domain=mesh, metadata={"quadrature_degree": 5})


def assemble(form):
    return dolfinx.fem.petsc.assemble_vector(qugar.dolfinx.form_custom(form)).array.copy()


cases = {
    "dsu only:  - q uh.n ds - q c.n ds":
        lambda w: -q * ufl.dot(uh, n) * ds - q * ufl.dot(w, n) * ds,
    "dx only:   q div(uh) dx + q c_0 dx":
        lambda w: q * ufl.div(uh) * dx + q * w[0] * dx,
    "dx + dsu:  q div(uh) dx - q uh.n ds - q c.n ds":
        lambda w: q * ufl.div(uh) * dx - q * ufl.dot(uh, n) * ds - q * ufl.dot(w, n) * ds,
    "dx + dsu:  -(-q div(uh) dx + q uh.n ds) - q c.n ds":
        lambda w: -(-q * ufl.div(uh) * dx + q * ufl.dot(uh, n) * ds) - q * ufl.dot(w, n) * ds,
}

for name, make in cases.items():
    with_constant, with_function = assemble(make(c)), assemble(make(c_fun))
    rel = np.linalg.norm(with_constant - with_function) / np.linalg.norm(with_function)
    print(f"{name:52s} rel. difference {rel:.2e}   |b| = {np.linalg.norm(with_function):.10e}")
