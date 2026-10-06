"""
stokes_infsup.py
Numerical inf-sup constant of an assembled Stokes CutFEM system.

Companion of compute_condition() in stokes_problem.py: it takes the same
SystemData and returns one stability number for the pressure side, so both
studies can run in the same loop on the same systems.

    coercivity side :  kappa(A_uu)                               (compute_condition)
    inf-sup side    :  beta^2 = lambda_min( B S^-1 B^T + C , T )  (compute_inf_sup)
                       kappa_p = lambda_max / lambda_min  of the same pencil

    B : discrete divergence (pressure-velocity block of sys.A, incl. Nitsche term)
    C : pressure block of sys.A (ghost penalty + global CIP, exactly as solved)
    S : velocity norm  mu ||grad v||^2_{Omega_T} + (mu/h) ||v||^2_Gamma
    T : pressure norm  (1/mu) ||q||^2_{Omega_T}

The constant pressure (exact kernel for full Dirichlet BC) is removed by
restricting to T-orthogonal (zero-mean) pressures.

IMPORTANT: call BEFORE solve_system(sys); solve_system pins sys.A in place.
Serial only.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla


def _to_scipy(M):
    ai, aj, av = M.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=M.getSize())


def _assemble(form, unfitted: bool):
    """unfitted=True : qugar custom quadrature (physical domain / Gamma)
       unfitted=False: standard quadrature on all active cells (Omega_T),
                       same route as the ghost-penalty terms in build_system"""
    import dolfinx.fem
    import dolfinx.fem.petsc
    f = dolfinx.fem.form(form) if unfitted else dolfinx.fem.form.__wrapped__(form)
    M = dolfinx.fem.petsc.assemble_matrix(f)
    M.assemble()
    out = _to_scipy(M)
    M.destroy()
    return out


def norm_matrices(sys, mu):
    """S (sparse, velocity norm) and T (dense, pressure norm), both on Omega_T."""
    import ufl
    from qugar.dolfinx import dsu
    mesh = sys.unf_mesh
    u, v = ufl.TrialFunction(sys.V), ufl.TestFunction(sys.V)
    p, q = ufl.TrialFunction(sys.Q), ufl.TestFunction(sys.Q)
    dxT = ufl.dx(domain=mesh)
    S = (_assemble(mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dxT, unfitted=False)
         + _assemble((mu / sys.h) * ufl.inner(u, v) * dsu(domain=mesh), unfitted=True))
    T = _assemble((1.0 / mu) * p * q * dxT, unfitted=False)
    return (0.5 * (S + S.T)).tocsc(), T.toarray()


def inf_sup_from_matrices(B, C, S, T, chunk=400, zero_tol=1e-9):
    """
    Pure linear algebra (no FE objects).
    B: sparse (nQ x nV), C: dense (nQ x nQ) or None, S: sparse SPD (nV x nV),
    T: dense SPD (nQ x nQ).
    """
    nQ = B.shape[0]

    # Schur complement B S^-1 B^T: one sparse LU of S, column-block solves
    lu = spla.splu(S)
    BT = B.T.tocsc()
    M = np.zeros((nQ, nQ))
    for j0 in range(0, nQ, chunk):
        j1 = min(j0 + chunk, nQ)
        M[:, j0:j1] = B @ lu.solve(BT[:, j0:j1].toarray())
    if C is not None:
        M = M + C
    M = 0.5 * (M + M.T)

    # Constant pressure: diagnostic + deflation
    one = np.ones(nQ)
    lam_const = float(one @ M @ one / (one @ T @ one))
    Z = sla.null_space((T @ one)[None, :])        # zero-mean pressures
    Mz = Z.T @ M @ Z
    Tz = Z.T @ T @ Z
    lam = sla.eigh(0.5 * (Mz + Mz.T), 0.5 * (Tz + Tz.T), eigvals_only=True)

    lam_min = lam[0]
    return dict(
        beta=float(np.sqrt(max(lam_min, 0.0))),
        kappa_p=float(lam[-1] / lam_min) if lam_min > zero_tol else np.inf,
        n_zero=int(np.sum(lam < zero_tol)),
        lam_const=lam_const,
    )


def compute_inf_sup(sys, mu=1.0, max_dofs_Q=6000):
    """
    Inf-sup constant of the system exactly as assembled by build_system.

    Returns dict with
        beta      : discrete inf-sup constant (stabilized by sys's own C)
        kappa_p   : lambda_max/lambda_min of the pressure Schur pencil
        n_zero    : number of spurious pressure modes (beyond the constant)
        lam_const : Rayleigh quotient of the constant pressure (should be ~0)
    or NaNs if the pressure space exceeds max_dofs_Q (dense eigenproblem).
    """
    nV, nQ = sys.n_dofs_V, sys.n_dofs_Q
    if nQ > max_dofs_Q:
        return dict(beta=np.nan, kappa_p=np.nan, n_zero=-1, lam_const=np.nan)

    A = _to_scipy(sys.A)
    B = A[nV:nV + nQ, :nV].tocsr()
    C = A[nV:nV + nQ, nV:nV + nQ].toarray()
    S, T = norm_matrices(sys, mu)
    return inf_sup_from_matrices(B, C, S, T)