"""Problem-independent CutFEM building blocks.

Everything here is shared by all problems: unfitted mesh creation, cut
measures, ghost penalty, assembly, linear solve, norms, linear algebra
diagnostics and VTK output. Problem modules only define their parameters,
exact solution and variational forms on top of these functions.

Two quadrature routes are used throughout (argument ``cut``):

* ``cut=True``: qugar custom quadrature, integrates over the physical domain
  Omega and the immersed boundary Gamma.
* ``cut=False``: standard quadrature over the full active cells Omega_T, used
  for ghost penalty and norms on the active mesh.
"""

from dataclasses import dataclass

import basix.ufl
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.mesh
import numpy as np
import qugar.dolfinx
import qugar.impl
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from qugar.dolfinx import UnfittedNormal, dsu
from qugar.mesh import create_unfitted_impl_Cartesian_mesh

DTYPE = dolfinx.default_scalar_type

# Mesh and measures


def disk_mesh(n_cells, xmin, xmax, center, radius, inside=True,
              comm=MPI.COMM_WORLD):
    """Create an unfitted Cartesian mesh cut by a disk.

    Parameters
    ----------
    n_cells : int or sequence of int
        Cells per direction (one int for a uniform count).
    xmin, xmax : array_like
        Corners of the background box.
    center, radius : array_like, float
        Immersed disk.
    inside : bool
        True: the physical domain is the disk. False: its complement in the
        box (flow around a cylinder).
    comm : MPI.Comm

    Returns
    -------
    mesh : qugar unfitted mesh
    h : float
        Largest cell side.
    """
    xmin = np.asarray(xmin, dtype=DTYPE)
    xmax = np.asarray(xmax, dtype=DTYPE)
    n = np.broadcast_to(np.asarray(n_cells), xmin.shape)
    h = float(np.max((xmax - xmin) / n))

    domain = qugar.impl.create_disk(radius, center=np.asarray(center, dtype=DTYPE))
    if not inside:
        domain = qugar.impl.create_negative(domain)

    mesh = create_unfitted_impl_Cartesian_mesh(
        comm, domain, n_cells, xmin, xmax, exclude_empty_cells=True, dtype=DTYPE)
    return mesh, h


def ghost_facets(mesh):
    """Interior facets with at least one cut neighbour cell (the set F_G)."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f2c = mesh.topology.connectivity(tdim - 1, tdim)

    starts = f2c.offsets[:-1]
    interior = np.flatnonzero(np.diff(f2c.offsets) == 2)
    cell_map = mesh.topology.index_map(tdim)
    is_cut = np.zeros(cell_map.size_local + cell_map.num_ghosts, dtype=bool)
    is_cut[mesh.get_cut_cells()] = True

    first = f2c.array[starts[interior]]
    second = f2c.array[starts[interior] + 1]
    return interior[is_cut[first] | is_cut[second]].astype(np.int32)


@dataclass(frozen=True)
class CutMeasures:
    """Integration measures and normals of an unfitted mesh."""

    dx: ufl.Measure        # physical domain Omega (cut) or Omega_T (standard)
    ds: ufl.Measure        # immersed boundary Gamma
    n: object              # unit normal on Gamma, pointing out of Omega
    dS: ufl.Measure        # all interior facets
    dS_ghost: ufl.Measure  # ghost-penalty facets F_G
    n_F: object            # facet normal on interior facets


def cut_measures(mesh):
    """Build the :class:`CutMeasures` of an unfitted mesh."""
    facets = ghost_facets(mesh)
    tags = dolfinx.mesh.meshtags(
        mesh, mesh.topology.dim - 1, facets, np.ones(len(facets), dtype=np.int32))
    return CutMeasures(
        dx=ufl.dx(domain=mesh),
        ds=dsu(domain=mesh),
        n=UnfittedNormal(mesh),
        dS=ufl.Measure("dS", domain=mesh),
        dS_ghost=ufl.Measure("dS", domain=mesh, subdomain_data=tags,
                             subdomain_id=1),
        n_F=ufl.FacetNormal(mesh),
    )


def lagrange_space(mesh, degree, shape=None):
    """Continuous Lagrange space; ``shape=(2,)`` for a vector field."""
    element = basix.ufl.element("Lagrange", mesh.basix_cell(), degree, shape=shape)
    return dolfinx.fem.functionspace(mesh, element)


# Ghost penalty


def normal_jump(w, n, order=1):
    """Jump of the ``order``-th normal derivative of ``w`` across a facet."""
    derivative = ufl.grad(w)
    for _ in range(order - 1):
        derivative = ufl.grad(derivative)

    n_plus = n("+")
    plus, minus = derivative("+"), derivative("-")
    for _ in range(order):
        plus, minus = ufl.dot(plus, n_plus), ufl.dot(minus, n_plus)
    return plus - minus


def ghost_penalty(u, v, measures, h, order=1):
    """Ghost penalty sum_{j=1}^{order} h^(2j-1) ([d_n^j u], [d_n^j v])_{F_G}.

    Scale by the physical coefficient outside (e.g. ``gamma * mu * ...``).
    """
    return sum(
        h ** (2 * j - 1) * ufl.inner(normal_jump(u, measures.n_F, j),
                                     normal_jump(v, measures.n_F, j))
        for j in range(1, order + 1)
    ) * measures.dS_ghost


# Assembly and solve


def compile_form(form, cut=True):
    """Compile a UFL form with qugar (``cut=True``) or standard quadrature."""
    return dolfinx.fem.form(form) if cut else dolfinx.fem.form.__wrapped__(form)


def assemble_matrix(form, cut=True, bcs=None, diag=1.0):
    """Assemble a bilinear form into a PETSc matrix."""
    A = dolfinx.fem.petsc.assemble_matrix(
        compile_form(form, cut), bcs=bcs or [], diag=diag)
    A.assemble()
    return A


def add_matrix(A, form, cut=False, bcs=None):
    """Assemble ``form`` and add it to ``A`` in place (typically ghost penalty)."""
    B = assemble_matrix(form, cut=cut, bcs=bcs, diag=0.0)
    A.axpy(1.0, B, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    B.destroy()


def assemble_vector(form, cut=True):
    """Assemble a linear form into a PETSc vector."""
    b = dolfinx.fem.petsc.assemble_vector(compile_form(form, cut))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    return b


def lu_solve(A, b):
    """Solve ``A x = b`` with a direct MUMPS LU factorization."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    x = A.createVecRight()
    ksp.solve(b, x)
    reason = ksp.getConvergedReason()
    ksp.destroy()
    if reason < 0:
        raise RuntimeError(f"LU solve failed (KSP reason {reason})")
    return x


def split_solution(x, *functions):
    """Copy a (monolithic) PETSc vector into Functions, block by block."""
    offset = 0
    for f in functions:
        index_map = f.function_space.dofmap.index_map
        n = index_map.size_local * f.function_space.dofmap.index_map_bs
        f.x.array[:n] = x.array_r[offset:offset + n]
        f.x.scatter_forward()
        offset += n


def join_solution(x, *functions):
    """Copy Functions into a (monolithic) PETSc vector; inverse of split_solution."""
    array, offset = x.array_w, 0
    for f in functions:
        index_map = f.function_space.dofmap.index_map
        n = index_map.size_local * f.function_space.dofmap.index_map_bs
        array[offset:offset + n] = f.x.array[:n]
        offset += n


@dataclass
class System:
    """Assembled CutFEM system; frees its PETSc objects when used in ``with``.

    Problem modules subclass this and add their spaces, measures and data.
    """

    mesh: object
    h: float
    A: PETSc.Mat
    b: PETSc.Vec

    @property
    def n_dofs(self):
        """Global number of unknowns."""
        return self.A.getSize()[0]

    def destroy(self):
        """Destroy every PETSc matrix and vector held by the system."""
        for value in vars(self).values():
            if isinstance(value, PETSc.Mat | PETSc.Vec):
                value.destroy()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


# Post-processing


def integrate(form):
    """Integrate a scalar form over the physical domain (summed over ranks)."""
    compiled = qugar.dolfinx.form_custom(form)
    local = dolfinx.fem.assemble_scalar(
        compiled, coeffs=compiled.pack_coefficients())
    return float(compiled.mesh.comm.allreduce(local, op=MPI.SUM))


def l2_norm(e, dx):
    """L2 norm of a (scalar or vector) expression."""
    return np.sqrt(integrate(ufl.inner(e, e) * dx))


def h1_seminorm(e, dx):
    """H1 seminorm of a (scalar or vector) expression."""
    return np.sqrt(integrate(ufl.inner(ufl.grad(e), ufl.grad(e)) * dx))


def eval_at_points(f, points):
    """Evaluate a Function at points; works in parallel.

    Each point is evaluated on a rank owning a cell that contains it. Points
    outside the active mesh give NaN.

    Parameters
    ----------
    f : dolfinx.fem.Function
    points : array_like, shape (n, 2) or (n, 3)

    Returns
    -------
    np.ndarray, shape (n,) for a scalar Function, (n, value_size) otherwise
    """
    import dolfinx.geometry

    mesh = f.function_space.mesh
    tdim = mesh.topology.dim
    points = np.atleast_2d(np.asarray(points, dtype=float))
    xyz = np.zeros((len(points), 3))
    xyz[:, :points.shape[1]] = points

    tree = dolfinx.geometry.bb_tree(mesh, tdim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, xyz)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, xyz)
    n_owned = mesh.topology.index_map(tdim).size_local

    idx, cells = [], []
    for i in range(len(xyz)):
        owned = [c for c in colliding.links(i) if c < n_owned]
        if owned:
            idx.append(i)
            cells.append(owned[0])

    value_size = int(np.prod(f.ufl_shape)) if f.ufl_shape else 1
    values = np.zeros((len(xyz), value_size))
    found = np.zeros(len(xyz))
    if idx:
        values[idx] = f.eval(xyz[idx], np.array(cells, dtype=np.int32))
        found[idx] = 1.0

    # A point on an edge between ranks is found twice: average
    comm = mesh.comm
    values = comm.allreduce(values, op=MPI.SUM)
    found = comm.allreduce(found, op=MPI.SUM)
    with np.errstate(invalid="ignore"):
        values /= found[:, None]
    return values[:, 0] if not f.ufl_shape else values


def mesh_geometry(mesh):
    """Cell and facet coordinates for plotting a cut mesh (serial).

    Returns a dict of plain arrays, so it can be stored with study data:
    ``full`` and ``cut`` cell boxes (n, 4) as (x0, y0, width, height), and
    ``ghost`` facet segments (n, 2, 2).
    """
    tdim = mesh.topology.dim
    x = mesh.geometry.x[:, :2]

    def boxes(cells):
        nodes = dolfinx.mesh.entities_to_geometry(mesh, tdim, cells)
        lo, hi = x[nodes].min(axis=1), x[nodes].max(axis=1)
        return np.hstack([lo, hi - lo])

    facets = ghost_facets(mesh)
    mesh.topology.create_connectivity(tdim - 1, tdim)
    segments = x[dolfinx.mesh.entities_to_geometry(mesh, tdim - 1, facets)]
    return {"full": boxes(mesh.get_full_cells()),
            "cut": boxes(mesh.get_cut_cells()),
            "ghost": segments}


# Linear algebra diagnostics (serial, dense)


def to_dense(A):
    """Dense NumPy copy of a PETSc matrix."""
    B = A.copy()
    dense = B.convert(PETSc.Mat.Type.DENSE).getDenseArray().copy()
    B.destroy()
    return dense


def to_scipy(A):
    """SciPy CSR copy of a (serial) PETSc matrix."""
    import scipy.sparse

    indptr, indices, data = A.getValuesCSR()
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=A.getSize())


def condition_number(A, max_dofs=50_000):
    """Spectral condition number sigma_max / sigma_min via a dense SVD.

    Returns NaN if the matrix is larger than ``max_dofs``.
    """
    if max(A.getSize()) > max_dofs:
        return np.nan
    sigma = np.linalg.svd(to_dense(A), compute_uv=False)
    return float(sigma[0] / sigma[-1])


def eigenvalues(A):
    """Ascending eigenvalues of a symmetric matrix."""
    return np.linalg.eigvalsh(to_dense(A))


class _InverseOperator:
    """Shell matrix A^{-1}: products are solves with one MUMPS LU of A."""

    def __init__(self, ksp):
        self.ksp = ksp

    def mult(self, mat, x, y):
        self.ksp.solve(x, y)

    def multTranspose(self, mat, x, y):
        self.ksp.solveTranspose(x, y)


def _largest_singular_value(M, tol):
    """Largest singular value of M by SLEPc thick-restart Lanczos."""
    from slepc4py import SLEPc

    svd = SLEPc.SVD().create(M.getComm())
    svd.setOperators(M)
    svd.setType(SLEPc.SVD.Type.TRLANCZOS)
    svd.setImplicitTranspose(True)
    svd.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
    svd.setDimensions(nsv=1)
    svd.setTolerances(tol=tol, max_it=1000)
    svd.solve()
    if svd.getConverged() < 1:
        raise RuntimeError("SLEPc SVD did not converge")
    sigma = svd.getValue(0)
    svd.destroy()
    return sigma


def condition_estimate(A, tol=1e-8):
    """2-norm condition number sigma_max / sigma_min of a large sparse matrix.

    Both extreme singular values come from SLEPc's Lanczos bidiagonalization
    (convergence decided on the residual). sigma_max from A; sigma_min as
    1 / sigma_max(A^{-1}), because the smallest singular values of A are
    clustered and converge slowly, while 1 / sigma_min is the well separated
    largest singular value of A^{-1}. A^{-1} is never formed: its products
    are solves with one MUMPS LU of A. Returns ``(kappa, sigma_max, sigma_min)``.
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    inverse = PETSc.Mat().createPython(A.getSizes(), _InverseOperator(ksp),
                                       comm=A.getComm())
    inverse.setUp()

    sigma_max = _largest_singular_value(A, tol)
    sigma_min = 1.0 / _largest_singular_value(inverse, tol)
    inverse.destroy()
    ksp.destroy()
    return sigma_max / sigma_min, sigma_max, sigma_min


def _lu_ksp(A):
    """KSP applying A^{-1} by a MUMPS LU factorization."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setUp()
    return ksp


def _set_lu(st):
    """Use a MUMPS LU for the linear solves of a SLEPc spectral transform."""
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")


class _SchurOperator:
    """Shell matrix K = A_up^T N^{-1} A_up + G_p (pressure Schur complement)."""

    def __init__(self, A_up, ksp_N, G_p):
        self.A_up, self.ksp_N, self.G_p = A_up, ksp_N, G_p
        self.t, self.w = A_up.createVecLeft(), A_up.createVecLeft()

    def mult(self, mat, x, y):
        self.A_up.mult(x, self.t)
        self.ksp_N.solve(self.t, self.w)
        self.A_up.multTranspose(self.w, y)
        self.G_p.multAdd(x, y, y)


def inf_sup_eigenvalues(N, A_up, G_p, M, tol=1e-10):
    """Extreme eigenvalues of (A_up^T N^{-1} A_up + G_p) q = lambda M q.

    The discrete inf-sup constant is beta_h = sqrt(lambda_min). Sparse
    version of the Chapelle & Bathe test, for meshes where the Schur
    complement cannot be formed:

    * lambda_min from the equivalent saddle point pencil
      [[N, A_up], [A_up^T, -G_p]] x = lambda [[0, 0], [0, -M]] x, by
      shift-and-invert (MUMPS LU) with a small negative target, so the
      shifted matrix is nonsingular even if lambda_min = 0;
    * lambda_max matrix-free, from the shell Schur operator (solves with N).

    N must be symmetric positive definite (Dirichlet rows: identity, with
    the matching rows of A_up zero). Returns ``(lambda_min, lambda_max)``.
    """
    from slepc4py import SLEPc

    comm = N.getComm()
    A_pu = A_up.copy()
    A_pu.transpose()
    G_neg = G_p.copy()
    G_neg.scale(-1.0)
    M_neg = M.copy()
    M_neg.scale(-1.0)
    zero = PETSc.Mat().createAIJ(N.getSizes(), nnz=0, comm=comm)   # empty block
    zero.assemble()
    lhs = PETSc.Mat().createNest([[N, A_up], [A_pu, G_neg]], comm=comm)
    rhs = PETSc.Mat().createNest([[zero, None], [None, M_neg]], comm=comm)
    lhs, rhs = lhs.convert("aij"), rhs.convert("aij")

    eps = SLEPc.EPS().create(comm)
    eps.setOperators(lhs, rhs)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setTarget(-1e-3)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    _set_lu(eps.getST())
    eps.setDimensions(nev=1)
    eps.setTolerances(tol=tol, max_it=500)
    eps.solve()
    values = [eps.getEigenvalue(i).real for i in range(eps.getConverged())]
    if not values:
        raise RuntimeError("SLEPc: no converged eigenvalue for lambda_min")
    lam_min = max(min(values), 0.0)
    for obj in (eps, lhs, rhs, A_pu, G_neg, M_neg, zero):
        obj.destroy()

    ksp_N = _lu_ksp(N)
    n_p = A_up.getSizes()[1]
    K = PETSc.Mat().createPython((n_p, n_p), _SchurOperator(A_up, ksp_N, G_p),
                                 comm=comm)
    K.setUp()
    eps = SLEPc.EPS().create(comm)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    _set_lu(eps.getST())
    eps.setDimensions(nev=1)
    eps.setTolerances(tol=tol, max_it=500)
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("SLEPc: no converged eigenvalue for lambda_max")
    lam_max = eps.getEigenvalue(0).real
    for obj in (eps, K, ksp_N):
        obj.destroy()
    return lam_min, lam_max


# Output


class VTKSeries:
    """Write Functions on a fitted reparametrization of the physical domain.

    The reparametrized mesh and the interpolation data are built once; each
    :meth:`write` interpolates the current values and adds a time to the
    ``.pvd`` file. Use as ``with VTKSeries(...) as vtk: vtk.write(t)``.

    Parameters
    ----------
    path : str or Path
        Output ``.pvd`` file; parent directories are created.
    mesh : qugar unfitted mesh
    functions : dict[str, dolfinx.fem.Function]
        Fields to write, by name.
    exact : dict[str, callable], optional
        ``exact[name](x)`` returns the exact UFL expression of field ``name``
        for spatial coordinates ``x``; adds ``<name>_exact`` and
        ``<name>_error`` fields.
    degree : int
        Polynomial degree of the reparametrized mesh and output spaces.
    """

    def __init__(self, path, mesh, functions, exact=None, degree=3):
        from pathlib import Path

        import dolfinx.io
        import qugar.reparam

        rep_mesh = qugar.reparam.create_reparam_mesh(
            mesh, degree=degree, levelset=False).create_mesh()
        x = ufl.SpatialCoordinate(rep_mesh)
        exact = exact or {}

        self._sources, self._errors, self._fields = [], [], []
        for name, f in functions.items():
            V = lagrange_space(rep_mesh, degree, shape=f.ufl_shape or None)
            f_rep = dolfinx.fem.Function(V, name=name)
            data = qugar.reparam.create_interpolation_data(V, f.function_space)
            self._sources.append((f_rep, f, data))
            self._fields.append(f_rep)

            if name in exact:
                f_ex = dolfinx.fem.Function(V, name=f"{name}_exact")
                f_ex.interpolate(dolfinx.fem.Expression(
                    exact[name](x), V.element.interpolation_points))
                f_err = dolfinx.fem.Function(V, name=f"{name}_error")
                self._errors.append((f_err, f_rep, f_ex))
                self._fields += [f_ex, f_err]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = dolfinx.io.VTKFile(rep_mesh.comm, str(path), "w")

    def write(self, t=0.0):
        """Interpolate the current values and write them at time ``t``."""
        for f_rep, f, data in self._sources:
            f_rep.interpolate_nonmatching(f, *data)
        for f_err, f_rep, f_ex in self._errors:
            f_err.x.array[:] = f_rep.x.array - f_ex.x.array
        self._file.write_function(self._fields, t)

    def close(self):
        """Close the output file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def save_vtk(path, mesh, functions, exact=None, degree=3):
    """Write Functions once; see :class:`VTKSeries` for the parameters."""
    with VTKSeries(path, mesh, functions, exact, degree) as vtk:
        vtk.write()
