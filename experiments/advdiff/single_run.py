"""Single advection-diffusion CutFEM run: end-to-end check and ParaView output."""

from cutfem import study
from cutfem.problems import advdiff

PARAMS = advdiff.Params(n_cells=128, mu=5e-3, gamma_beta=0.1)


def compute():
    """Solve once and write the solution for ParaView."""
    study.print0(PARAMS)
    with advdiff.build(PARAMS) as system:
        uh = advdiff.solve(system)
        path = study.RESULTS_DIR / "advdiff" / "single_run.pvd"
        advdiff.save_vtk(path, system, uh)
        values = uh.x.array
        summary = {"h": system.h, "n_dofs": system.n_dofs,
                   "Pe_h": advdiff.peclet_h(PARAMS, system.h),
                   "min_uh": float(values.min()), "max_uh": float(values.max())}
    for key, value in summary.items():
        study.print0(f"  {key:>7} = {value:.4g}")
    study.print0(f"Saved {path.relative_to(study.ROOT)}")
    return {"params": PARAMS, **summary}


if __name__ == "__main__":
    study.run("advdiff/single_run", compute)
