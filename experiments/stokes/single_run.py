"""Single Stokes CutFEM run: end-to-end check and ParaView output."""

from cutfem import study
from cutfem.problems import stokes

PARAMS = stokes.Params(n_cells=64, gamma_mu=10.0, gamma_g=0.1, gamma_p=1.0,
                       v_degree=1, p_degree=1)


def compute():
    """Solve once, print the errors and write the solution for ParaView."""
    study.print0(PARAMS)
    with stokes.build(PARAMS) as system:
        solution = stokes.solve(system)
        errors = stokes.errors(system, solution)
        path = study.RESULTS_DIR / "stokes" / "single_run.pvd"
        stokes.save_vtk(path, system, solution)
        study.print0(f"h = {system.h:.5f}, n_dofs = {system.n_dofs}")
    for key, value in errors.items():
        study.print0(f"  {key:>6} = {value:.4e}")
    study.print0(f"Saved {path.relative_to(study.ROOT)}")
    return {"params": PARAMS, "errors": errors}


if __name__ == "__main__":
    study.run("stokes/single_run", compute)
