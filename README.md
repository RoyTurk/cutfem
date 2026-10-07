# cutfem

CutFEM numerical experiments (Poisson, Stokes, advection-diffusion,
Navier-Stokes) with dolfinx and qugar.

## Layout

```
src/cutfem/
    core.py             mesh, measures, ghost penalty, assembly, solve, norms, VTK
    study.py            sweeps, tables, rates, data/figure output, command line
    plotting.py         thesis figure sizes, colours and helpers
    thesis.mplstyle     the style of every figure
    problems/           one module per problem: Params, build, solve, errors
experiments/<problem>/  study scripts
results/                data (.json) and ParaView output (git-ignored)
figures/                figures (.pdf) for the thesis (git-ignored)
```

## Setup

```bash
conda activate qugar-env
pip install -e . --no-deps
```

## Running a study

```bash
python experiments/poisson/convergence.py              # compute, save data and figure
python experiments/poisson/convergence.py --plot-only  # redraw from saved data
python experiments/poisson/convergence.py --show --formats pdf png
```

Data goes to `results/<problem>/<study>.json` (with parameters, git commit and
date), figures to `figures/<problem>/<study>.pdf`. Figures are made at their
printed size: include them in LaTeX without scaling,
`\includegraphics{figures/poisson/convergence.pdf}`.

## Known issue: JIT linking

The linker in `qugar-env` is older than the macOS SDK, so compiling new forms
fails at the link step. Until the environment is updated, use the system
compiler:

```bash
export CC=/usr/bin/clang
```
