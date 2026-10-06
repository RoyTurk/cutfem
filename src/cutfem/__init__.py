"""CutFEM numerical experiments.

Modules
-------
core
    Problem-independent building blocks: mesh, measures, ghost penalty,
    assembly, solve, norms, linear algebra diagnostics, VTK output.
study
    Running studies: parameter sweeps, tables, convergence rates, and the
    ``run`` entry point that saves data and figures.
plotting
    Thesis figure style, sizes, palettes and plot helpers.
problems
    One module per PDE: parameters, exact solution, variational forms.
"""
