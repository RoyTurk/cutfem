# Newton's method for the stationary CutFEM Navier–Stokes problem

This note derives the Newton linearisation of the stationary CutFEM
formulation and shows what is solved at each iteration, both as variational
forms and as block matrices.

Notation is the same as in the formulation:
$a_h$ (viscous + Nitsche), $b_h$ (pressure coupling), $c$ (convection),
$g_u, g_p$ (ghost penalties), $\ell_u, \ell_p$ (right-hand sides).

---

## 1. The discrete problem as a residual

Find $(\mathbf u_h, p_h) \in V_h \times Q_h$, with $\mathbf u_h = \mathbf g_D$ on
$\Gamma_D^s$, such that $R(\mathbf u_h, p_h; \mathbf v, q) = 0$ for all
$(\mathbf v, q) \in V_{h,0} \times Q_h$, where

$$
\begin{aligned}
R_u(\mathbf u, p; \mathbf v) &= a_h(\mathbf u, \mathbf v) + c(\mathbf u, \mathbf u, \mathbf v)
  + b_h(\mathbf v, p) + g_u(\mathbf u, \mathbf v) - \ell_u(\mathbf v), \\
R_p(\mathbf u, p; q) &= -\,b_h(\mathbf u, q) + g_p(p, q) - \ell_p(q),
\end{aligned}
$$

and $R = R_u + R_p$.

**Every term is linear in $(\mathbf u, p)$ except $c(\mathbf u, \mathbf u, \mathbf v)$**,
which is quadratic in $\mathbf u$. So the nonlinearity, and all the work in
this note, comes from that one term.

> This assumes the ghost-penalty weight $\phi_u = \nu + U^2h^2/(\nu+Uh)$ uses a
> **fixed** characteristic velocity $U$ (a parameter). If $U$ depended on
> $\mathbf u_h$, $g_u$ and $g_p$ would be nonlinear too.

---

## 2. Idea of Newton's method

Given an iterate $(\mathbf u_k, p_k)$, look for a correction
$(\delta\mathbf u, \delta p)$ such that
$(\mathbf u_{k+1}, p_{k+1}) = (\mathbf u_k + \delta\mathbf u,\ p_k + \delta p)$
solves the problem better. Expand the residual to first order in the correction
and set that expansion to zero:

$$
R(\mathbf u_k + \delta\mathbf u, p_k + \delta p) \approx
R(\mathbf u_k, p_k) + DR(\mathbf u_k, p_k)[\delta\mathbf u, \delta p] = 0 .
$$

$DR(\mathbf u_k, p_k)[\delta\mathbf u, \delta p]$ is the (Gâteaux) derivative
of $R$ at $(\mathbf u_k, p_k)$ in the direction $(\delta\mathbf u, \delta p)$:

$$
DR(\mathbf u_k, p_k)[\delta\mathbf u, \delta p] =
\frac{d}{d\varepsilon} R(\mathbf u_k + \varepsilon\,\delta\mathbf u,\ p_k + \varepsilon\,\delta p)\Big|_{\varepsilon = 0}.
$$

---

## 3. Derivative of each term

**Linear terms.** For any linear form $L$, the derivative is
$L$ itself applied to the correction. For example,
$\frac{d}{d\varepsilon} a_h(\mathbf u_k + \varepsilon\delta\mathbf u, \mathbf v)|_0 = a_h(\delta\mathbf u, \mathbf v)$.
The same holds for $b_h$, $g_u$ and $g_p$. The data terms $\ell_u, \ell_p$ do not
depend on the unknowns, so their derivative is zero.

**Convective term.** $c$ is linear in each of its first two arguments, so

$$
c(\mathbf u_k + \varepsilon\delta\mathbf u,\ \mathbf u_k + \varepsilon\delta\mathbf u,\ \mathbf v)
= c(\mathbf u_k, \mathbf u_k, \mathbf v)
+ \varepsilon\big[\,c(\mathbf u_k, \delta\mathbf u, \mathbf v) + c(\delta\mathbf u, \mathbf u_k, \mathbf v)\,\big]
+ \varepsilon^2\, c(\delta\mathbf u, \delta\mathbf u, \mathbf v).
$$

Differentiating at $\varepsilon = 0$:

$$
Dc(\mathbf u_k)[\delta\mathbf u] =
\underbrace{c(\mathbf u_k, \delta\mathbf u, \mathbf v)}_{\text{(i)}}
+ \underbrace{c(\delta\mathbf u, \mathbf u_k, \mathbf v)}_{\text{(ii)}} .
$$

* (i) $\int_\Omega (\mathbf u_k\cdot\nabla)\,\delta\mathbf u\cdot\mathbf v$:
  the correction is transported by the current velocity (an Oseen term).
* (ii) $\int_\Omega (\delta\mathbf u\cdot\nabla)\,\mathbf u_k\cdot\mathbf v$:
  the current velocity gradient acts on the correction like a reaction term.
  Picard iteration drops this term (Section 6).

The only term dropped by the linearisation is
$c(\delta\mathbf u, \delta\mathbf u, \mathbf v)$, which is quadratic in the
correction. That is why Newton converges quadratically close to the solution:
the error after a step is of the order of the square of the error before it.

---

## 4. One Newton step

**Correction form.** Find $(\delta\mathbf u, \delta p) \in V_{h,0}\times Q_h$ such that,
for all $(\mathbf v, q) \in V_{h,0}\times Q_h$,

$$
\boxed{
\begin{aligned}
a_h(\delta\mathbf u, \mathbf v) + c(\mathbf u_k, \delta\mathbf u, \mathbf v) + c(\delta\mathbf u, \mathbf u_k, \mathbf v)
+ b_h(\mathbf v, \delta p) + g_u(\delta\mathbf u, \mathbf v) &= -R_u(\mathbf u_k, p_k; \mathbf v), \\
-\,b_h(\delta\mathbf u, q) + g_p(\delta p, q) &= -R_p(\mathbf u_k, p_k; q),
\end{aligned}}
$$

then update $\mathbf u_{k+1} = \mathbf u_k + \delta\mathbf u$, $p_{k+1} = p_k + \delta p$.

**Equivalent form for the new iterate.** Substitute
$\delta\mathbf u = \mathbf u_{k+1} - \mathbf u_k$ and $\delta p = p_{k+1} - p_k$.
All linear terms combine, and the convective terms give
$c(\mathbf u_k, \mathbf u_k) + c(\mathbf u_k, \delta\mathbf u) + c(\delta\mathbf u, \mathbf u_k)
= c(\mathbf u_k, \mathbf u_{k+1}) + c(\mathbf u_{k+1}, \mathbf u_k) - c(\mathbf u_k, \mathbf u_k)$.
Hence: find $(\mathbf u_{k+1}, p_{k+1})$, with $\mathbf u_{k+1} = \mathbf g_D$ on $\Gamma_D^s$, such that

$$
\begin{aligned}
a_h(\mathbf u_{k+1}, \mathbf v) + c(\mathbf u_k, \mathbf u_{k+1}, \mathbf v) + c(\mathbf u_{k+1}, \mathbf u_k, \mathbf v)
+ b_h(\mathbf v, p_{k+1}) + g_u(\mathbf u_{k+1}, \mathbf v) &= \ell_u(\mathbf v) + c(\mathbf u_k, \mathbf u_k, \mathbf v), \\
-\,b_h(\mathbf u_{k+1}, q) + g_p(p_{k+1}, q) &= \ell_p(q).
\end{aligned}
$$

Both forms give the same iterates. The correction form is preferred in code:
the right-hand side is the residual, whose norm is also the stopping criterion,
and the correction has homogeneous boundary conditions.

### Boundary conditions

* **Strong ($\Gamma_D^s$).** Choose $\mathbf u_0$ with $\mathbf u_0 = \mathbf g_D$ on
  $\Gamma_D^s$ (the Stokes solution has this). Every correction lies in
  $V_{h,0}$, i.e. $\delta\mathbf u = \mathbf 0$ on $\Gamma_D^s$, so every iterate
  keeps the inlet and wall values.
* **Nitsche ($\Gamma$).** Nothing special: the Nitsche terms are part of
  $a_h$ and $\ell_u$, so they appear in the residual and are linear.
* **Neumann ($\Gamma_N$).** $\mathbf g_N$ is data in $\ell_u$; for do-nothing it is zero.

---

## 5. Matrix form

Let $\{\boldsymbol\varphi_j\}$ be the velocity basis and $\{\psi_j\}$ the pressure basis,
and $\mathbf u_k = \sum_j U^k_j \boldsymbol\varphi_j$. The blocks are

| Block | Entry $(i, j)$ | Depends on $\mathbf u_k$? |
|---|---|---|
| $A$ | $a_h(\boldsymbol\varphi_j, \boldsymbol\varphi_i) + g_u(\boldsymbol\varphi_j, \boldsymbol\varphi_i)$ | no, assemble once |
| $B_{up}$ | $b_h(\boldsymbol\varphi_i, \psi_j)$ | no |
| $B_{pu}$ | $-\,b_h(\boldsymbol\varphi_j, \psi_i)$ | no |
| $G_p$ | $g_p(\psi_j, \psi_i)$ | no |
| $N(\mathbf u_k)$ | $c(\mathbf u_k, \boldsymbol\varphi_j, \boldsymbol\varphi_i) = \int_\Omega (\mathbf u_k\cdot\nabla)\boldsymbol\varphi_j\cdot\boldsymbol\varphi_i$ | yes |
| $W(\mathbf u_k)$ | $c(\boldsymbol\varphi_j, \mathbf u_k, \boldsymbol\varphi_i) = \int_\Omega (\boldsymbol\varphi_j\cdot\nabla)\mathbf u_k\cdot\boldsymbol\varphi_i$ | yes |

The residual vector at $(U^k, P^k)$ is

$$
\mathbf r_k =
\begin{bmatrix} \mathbf r_u \\ \mathbf r_p \end{bmatrix}
=
\begin{bmatrix} A + N(\mathbf u_k) & B_{up} \\ B_{pu} & G_p \end{bmatrix}
\begin{bmatrix} U^k \\ P^k \end{bmatrix}
-
\begin{bmatrix} \mathbf f_u \\ \mathbf f_p \end{bmatrix},
$$

using $N(\mathbf u_k)\,U^k \leftrightarrow c(\mathbf u_k, \mathbf u_k, \cdot)$.
The Newton system (Jacobian) is

$$
\underbrace{\begin{bmatrix} A + N(\mathbf u_k) + W(\mathbf u_k) & B_{up} \\ B_{pu} & G_p \end{bmatrix}}_{J_k}
\begin{bmatrix} \delta U \\ \delta P \end{bmatrix}
= -\,\mathbf r_k .
$$

Compared with Stokes, only the velocity–velocity block changes: it gains
$N + W$ and stops being symmetric. The other three blocks are exactly the Stokes blocks
(with $\mu \to \nu$ and the new ghost-penalty weights), so only $N$ and $W$
have to be reassembled at each iteration (cut quadrature on $\Omega$).
Rows of strongly constrained dofs on $\Gamma_D^s$ are replaced by identity rows
with zero right-hand side.

---

## 6. Comparison with Picard

Picard (Oseen fixed point) freezes the transporting velocity and drops (ii):

$$
a_h(\mathbf u_{k+1}, \mathbf v) + c(\mathbf u_k, \mathbf u_{k+1}, \mathbf v) + b_h(\mathbf v, p_{k+1}) + g_u(\mathbf u_{k+1}, \mathbf v) = \ell_u(\mathbf v),
\qquad \text{(continuity unchanged)}.
$$

In matrix form, $J_k$ is replaced by the same matrix without $W(\mathbf u_k)$.

| | Picard | Newton |
|---|---|---|
| Velocity block | $A + N(\mathbf u_k)$ | $A + N(\mathbf u_k) + W(\mathbf u_k)$ |
| Convergence | linear, contraction factor grows with Re | quadratic near the solution |
| Starting guess | converges from far away (small-data regime) | needs a guess close enough to the solution |
| Each iteration | one Oseen solve | one solve of the same size and sparsity |

Both cost one linear solve of the same size per iteration. Newton just needs far fewer of them.
Hybrid: a few Picard steps to get close, then Newton.

---

## 7. Algorithm

```
assemble A, B_up, B_pu, G_p, f_u, f_p          # once
(u_0, p_0) = Stokes solution                    # satisfies g_D on Γ_D^s
optionally: n_picard Picard steps
for k = 0, 1, 2, ...
    assemble N(u_k), W(u_k)                      # cut quadrature on Ω
    r_k = residual(u_k, p_k)                     # constrained rows set to 0
    if ||r_k|| <= rtol * ||r_0|| + atol: stop
    solve J_k [δU; δP] = -r_k                    # MUMPS LU
    U += α δU,  P += α δP                        # α = 1, or damped
    if ||δU|| + ||δP|| small: stop
```

Without a Neumann boundary ($\Gamma_N = \emptyset$), the pressure in each
correction is defined up to a constant. Pin one pressure dof of $\delta p$ to zero,
and shift $p$ to zero mean after convergence.

Typical residual history near the solution (illustrative only): the exponent
roughly doubles at each step, e.g. $10^{-1}, 10^{-3}, 10^{-7}, 10^{-14}$.
Linear (Picard) convergence instead reduces the residual by a roughly constant factor per step.

---

## 8. In UFL

With $(\mathbf w\cdot\nabla)\mathbf z = $ `ufl.dot(ufl.grad(z), w)`:

```python
def convection(w, z, v, dx):
    """c(w, z, v) = ∫ (w·∇)z · v."""
    return ufl.inner(ufl.dot(ufl.grad(z), w), v) * dx

# u_k: dolfinx Function (current iterate), du: TrialFunction, v: TestFunction
N_form = convection(u_k, du, v, m.dx)          # Picard part
W_form = convection(du, u_k, v, m.dx)          # extra Newton part
r_conv = convection(u_k, u_k, v, m.dx)         # convective part of the residual
```

`ufl.derivative(r_conv, u_k, du)` produces exactly `N_form + W_form`, which is a quick
check that the hand-derived Jacobian is right.
