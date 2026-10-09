"""Single unsteady Navier-Stokes CutFEM run (DFG 2D-3, Re <= 100).

Time loop on [0, t_end] with BDF2 and a Newton solve per step. Saves the
drag and lift coefficients at every step to results/dfg2d3/single_run.json
(also every ``--save-every`` steps during the run, so a long run can be
inspected or recovered), velocity and pressure every ``--vtk-every`` time
units to a ParaView time series, and plots c_D(t), c_L(t) against the
FEATFLOW reference (Q2/P1disc level 6, Crank-Nicolson, dt = 1/1600).

Usage::

    python single_run.py                          # n_y = 40, dt = 1/1600
    python single_run.py --dt 0.02                # quick run on [0, 8]
    python single_run.py --plot-only
"""

import time

from cutfem import plotting, study
from cutfem.problems import dfg2d3

NAME = "dfg2d3/single_run"
REFERENCE = (study.ROOT / "data" / "featflow" / "draglift_q2_cn_lv1-6_dt4"
             / "bdforces_lv6")


def arguments(parser):
    """Mesh, time step and output options from the command line."""
    defaults = dfg2d3.Params()
    parser.add_argument("--n-y", type=int, default=defaults.n_y,
                        help="cells across the channel (h = 0.41 / n_y)")
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--t-end", type=float, default=defaults.t_end)
    parser.add_argument("--save-every", type=int, default=400,
                        help="steps between intermediate data saves")
    parser.add_argument("--vtk-every", type=float, default=0.1,
                        help="time between ParaView snapshots")


def compute(args):
    """Run the time loop, report progress and save intermediate data."""
    params = dfg2d3.Params(n_y=args.n_y, dt=args.dt, t_end=args.t_end)
    n_steps = round(params.t_end / params.dt)
    vtk_steps = max(1, round(args.vtk_every / params.dt))
    vtk_path = study.RESULTS_DIR / "dfg2d3" / f"solution_ny{params.n_y}.pvd"
    start = time.perf_counter()

    with dfg2d3.build(params) as system, dfg2d3.vtk_series(vtk_path, system) as vtk:
        study.print0(params)
        study.print0(f"h = {system.h:.4e}, n_dofs = {system.n_dofs}, "
                     f"steps = {n_steps}, MPI ranks = {system.mesh.comm.size}")
        table = study.Table(["step", "t", "newton", "c_D", "c_L", "elapsed"])
        table.header()

        def on_step(k, record):
            t = record["t"][-1]
            if k % vtk_steps == 0 or k == n_steps:
                vtk.write(t)
            if k > 0 and (k % args.save_every == 0 or k == n_steps):
                table.row([k, t, record["newton"][-1], record["c_D"][-1],
                           record["c_L"][-1], time.perf_counter() - start])
                if system.mesh.comm.rank == 0:
                    study.save_data(NAME, {"params": params, **record})

        record = dfg2d3.solve(system, on_step=on_step)
        dp_end = dfg2d3.pressure_drop(system)

    values = {**dfg2d3.peaks(record), "dp_end": dp_end}
    study.print0(f"\n  {'quantity':<10}{'computed':>14}{'reference':>14}")
    for key, ref in dfg2d3.DFG_2D3_REF.items():
        study.print0(f"  {key:<10}{values[key]:>14.6f}{ref:>14.6f}")
    study.print0(f"Saved {vtk_path.relative_to(study.ROOT)}")
    return {"params": params, **record, **values,
            "runtime": time.perf_counter() - start}


def plot(data):
    """c_D(t) and c_L(t) against the FEATFLOW reference."""
    reference = dfg2d3.load_reference(REFERENCE)
    fig, axes = plotting.figure(nrows=2, aspect=0.6)
    labels = {"c_D": r"$c_D$", "c_L": r"$c_L$"}
    for ax, key in zip(axes, labels, strict=True):
        ax.plot(reference["t"], reference[key], color=plotting.INK,
                linestyle="--", linewidth=0.8, label="FEATFLOW level 6")
        ax.plot(data["t"], data[key],
                label=rf"CutFEM, $\Delta t = {data['params']['dt']:.3g}$")
        ax.set_xlim(0.0, 8.0)
        ax.set_ylabel(labels[key])
    axes[-1].set_xlabel(r"$t$")
    plotting.legend_above(fig)
    return fig


if __name__ == "__main__":
    study.run(NAME, compute, plot, arguments=arguments)
