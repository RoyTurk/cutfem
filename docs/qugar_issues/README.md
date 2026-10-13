# Two wrong-result issues in qugar's runtime-quadrature forms

Two independent problems in forms compiled by `qugar.dolfinx` (custom cut-cell
and unfitted-boundary quadrature). Both make assembly return **wrong values
without any error or warning**. Each one has a standalone reproducer in this
folder that uses only DOLFINx and qugar.

| | Issue 1 | Issue 2 |
|---|---|---|
| Reproducer | `repro_quadrature_order.py` | `repro_p1_coordinate.py` |
| Symptom | wrong values or NaN | wrong values (errors of 4–25 %) |
| Trigger | one form with `dx` + `dsu` integrals, a `Function` and a `Constant` | P1 `Function` × expression of `x[1]` in a cut-cell `dx` integral |
| Deterministic? | no: depends on `PYTHONHASHSEED` | yes |
| Workaround | compile the `dx` and `dsu` parts as separate forms | interpolate coordinate expressions into a Lagrange `Function` |

## Environment

| | |
|---|---|
| qugar | 0.5.0 |
| DOLFINx / basix / FFCx | 0.11.0 |
| UFL | 2026.1.0 |
| PETSc | 3.25.5 |
| Python | 3.11.15 |
| Platform | macOS 27.0.1, arm64 (conda environment) |

All runs are serial. The meshes come from `create_unfitted_impl_Cartesian_mesh`
(quadrilaterals) with a disk, or the complement of a disk, as the implicit domain.

---

## Issue 1: forms mixing `dx` and `dsu` give wrong values or NaN, depending on `PYTHONHASHSEED`

### Description

A form can assemble to wrong values, or to NaN, when it contains all of:

1. a cut-cell integral (`ufl.dx`) **and** an unfitted-boundary integral
   (`qugar.dolfinx.dsu`) in the same form;
2. a `Function` coefficient;
3. a `Constant`.

Which variant of the form fails changes with `PYTHONHASHSEED`. Two
mathematically identical ways of writing the same form can fail for different
seeds. For each seed tested, some variants were correct and others were not.

### Reproduction

```bash
for s in 0 1 2 3; do PYTHONHASHSEED=$s python repro_quadrature_order.py; done
```

The domain is the unit square minus a disk on a 16 × 16 mesh, with P2 vector
`uh` (random values), P1 test function `q` and a vector `Constant` c = (0.3, −0.2).
Each form is assembled twice: once with the `Constant` c, and once with a
`Function` interpolating the same constant value. The second version is
always consistent and serves as the reference. The two should agree to
round-off.

Relative difference ‖b_Constant − b_Function‖ / ‖b_Function‖:

| Form | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| `- q uh·n dsu - q c·n dsu` (dsu only) | 1e-16 | 1e-16 | 1e-16 | 1e-16 |
| `q div(uh) dx + q c₀ dx` (dx only) | 0 | 0 | 0 | 0 |
| `q div(uh) dx - q uh·n dsu - q c·n dsu` | **NaN** | 5e-17 | 5e-17 | 5e-17 |
| `-(-q div(uh) dx + q uh·n dsu) - q c·n dsu` (same form) | 5e-17 | **0.68** | **0.68** | **0.68** |

### What we observed

- **Forms with a single measure** (only `dx`, or only `dsu`) were always correct.
- **No `Function`:** the bilinear analogue of the same form, with a `TrialFunction`
  instead of `uh` and a `Constant` factor, was always correct.
- **No `Constant`:** with the `Constant` replaced by a `Function`, or without the
  `Constant` term, the forms were always correct.
- **A `Constant` inside the same product as `uh`** (`- q (uh + c)·n dsu`) was
  correct for the seeds tested. The failures needed the `Constant` in a term of
  its own.
- **Splitting the form** into its `dx` integrals and its `dsu` integrals,
  compiling the two as separate forms and adding the assembled vectors, gives
  the correct result for every seed.
- **Determinism:** for a fixed seed the result is reproducible, and different
  seeds give different results. So some ordering inside form compilation or
  coefficient packing depends on string hashing. The same form gives identical
  integrals in UFL (`form.integrals()` and their metadata print identically in
  the correct and incorrect cases).

We have not identified the root cause. The pattern (a kernel holding two
quadrature rules, plus extra constant and coefficient data) suggests the order
in which per-cell custom data, coefficients and constants are packed does not
always match the order in which the generated kernel reads them. NaN results
would fit reading from the wrong offset.

### How we found it

In a Navier–Stokes solver (Newton's method, residual and Jacobian forms
containing the current iterate as a `Function`), Newton converged quadratically
on some meshes and diverged on others. A finite-difference check showed the
Jacobian did not match the residual. The mismatch came and went with
`PYTHONHASHSEED`.

### Workarounds

Either of these removes the failure in our solver for every seed tested:

- **No `Constant` in cut forms that contain a `Function`.** Absent data terms
  (zero body force, no-slip data) are left out of the form instead of
  being written with a zero `Constant`. This is what the solver uses now.
- **Split every cut form** into its `dx` part and its `dsu` part
  (`metadata["custom_unfitted_boundary"]` tells them apart), compile each part
  separately, and add the assembled matrices or vectors.

How fragile the trigger is shows up in our solver's continuity residual
`-b_h(uh, q) ∓ q (g·n) dsu`, with g a zero `Constant` (no-slip). With the
`-` sign, which was a sign error on our side (mathematically irrelevant for
g = 0), the residual was wrong for `PYTHONHASHSEED` = 1, 2 and 3: the
Jacobian was 12 % off and Newton diverged. With the `+` sign, the same form
was correct for all seeds. Both workarounds fix the `-` version.

---

## Issue 2: P1 Function times an expression of `x[1]` integrates incorrectly on cut cells

### Description

The cut-cell integral of `f * g(x)` is wrong when `f` is a **P1** (Q1)
`Function` and `g` is an expression of `ufl.SpatialCoordinate` containing
`x[1]`. P2 Functions are correct, and so are most `x[0]`-only expressions.
Cells without a cut are correct. The result does not depend on `PYTHONHASHSEED`.

### Reproduction

```bash
python repro_p1_coordinate.py
```

The domain is a disk of radius 0.5 centred at (0.03, 0.01), on a 16 × 16 mesh
of [−1, 1]². `f = sin(2x) + y` is interpolated into P1, and `f6` is an exact
copy of `f` in P6 (as P1 ⊂ P6). Both integrals should agree up to quadrature
error (about 1e-7):

| g | ∫ f g (f in P1) | ∫ f6 g (exact copy in P6) | rel. difference |
|---|---|---|---|
| x₀ | 0.0910881979 | 0.0910881803 | 2e-7 |
| x₀² | 0.0081509507 | 0.0081509461 | 6e-7 |
| x₀ x₁ | 0.0023835038 | 0.0023835034 | 2e-7 |
| **x₁** | 0.0464161542 | 0.0495784656 | **6.4e-2** |
| **x₁²** | 0.0031391620 | 0.0041690444 | **2.5e-1** |
| **x₀² x₁** | 0.0023420629 | 0.0021709958 | **7.9e-2** |
| **sin(x₁)** | 0.0467437720 | 0.0485456125 | **3.7e-2** |

With `f` in P2 instead, every row agrees to 1e-7 or better.

### What we observed

- **Only P1, and only expressions involving `x[1]`.** `x[0]`, `x[0]**2` and
  `x[0]*x[1]` were correct, while `x[1]`, `x[1]**2`, `x[0]**2*x[1]` and
  `sin(x[1])` were wrong.
- **Only cut cells.** On a mesh whose cells are all full (a disk containing the
  whole box), the same integrals agree to 1e-15. So the problem is in the
  runtime-quadrature kernel, not the standard one.
- **It depends on how the expression is written.** For example,
  `inner((f - a) - (p - b), (f - a) - (p - b))` (with float constants a, b) was
  correct, while the mathematically equal `inner(f - p + (b - a), …)` was wrong
  by 30 %. Here p = 60 x₀² x₁ − 20 x₁³.
- **Same in vector forms:** a P1 `Function` and a polynomial of `x` appearing
  together in one residual gave a Jacobian inconsistent with the residual.

The mesh's coordinate element is also Q1. A possible explanation is that the
P1 basis table and the coordinate-element table are merged, or confused, in the
runtime-quadrature code path, with the second component of `x` read at the
wrong offset. This is a hypothesis, not a verified diagnosis.

### Workaround in use

Data that depends on `x` (body forces, boundary data, exact solutions in error
norms) is interpolated into a higher-degree Lagrange `Function` (P_{k+2}, or P6
for error norms) before it enters a cut form, so no cut kernel contains a
`SpatialCoordinate`.

---

## Files

- `repro_quadrature_order.py`: issue 1 (run with several `PYTHONHASHSEED` values).
- `repro_p1_coordinate.py`: issue 2.

Both scripts only need DOLFINx and qugar. On our machine the JIT link step
required `export CC=/usr/bin/clang` (an unrelated toolchain mismatch in the
conda environment).
