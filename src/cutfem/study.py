"""Running studies: sweeps, tables, convergence rates, data and figure output.

A study script defines ``compute() -> data`` and ``plot(data) -> figure(s)``
and calls ``run(name, compute, plot)``. Results and figures go to::

    results/<name>.json      data + metadata (parameters, git commit, date)
    figures/<name>.pdf       figure(s)

Command line options of every study::

    --plot-only     redraw the figures from saved data, no computation
    --show          open the figures in a window
    --formats ...   figure formats (default: pdf)
"""

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def print0(*args, **kwargs):
    """Print on MPI rank 0 only."""
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs, flush=True)


# Tables and rates


class Table:
    """Fixed-width table printed row by row while a study runs."""

    def __init__(self, columns, width=12):
        self.columns = list(columns)
        self.width = width

    def header(self):
        """Print the column names and a rule."""
        print0("  " + " ".join(f"{c:>{self.width}}" for c in self.columns))
        print0("  " + "-" * ((self.width + 1) * len(self.columns) - 1))

    def row(self, values):
        """Print one row from a dict (or sequence) of values."""
        if isinstance(values, dict):
            values = [values[c] for c in self.columns]
        print0("  " + " ".join(self._format(v) for v in values))

    def _format(self, value):
        if isinstance(value, int | np.integer):
            return f"{value:>{self.width}d}"
        if isinstance(value, float | np.floating):
            return f"{value:>{self.width}.4e}"
        if isinstance(value, tuple | list | np.ndarray):
            value = "(" + ", ".join(f"{v:.3g}" for v in value) + ")"
        return f"{value!s:>{self.width}}"


def rates(x, y):
    """Observed orders log(y_i / y_{i+1}) / log(x_i / x_{i+1})."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(y[1:] / y[:-1]) / np.log(x[1:] / x[:-1])


def print_rates(columns, expected, x="h"):
    """Print observed convergence orders next to the expected ones."""
    print0("  Observed orders:")
    for key, order in expected.items():
        observed = "  ".join(f"{r:6.2f}" for r in rates(columns[x], columns[key]))
        print0(f"  {key:>8}: {observed}   (expected {order:.2f})")


# Sweeps


def sweep(problem, params, vary, values, extras=None):
    """Run ``problem`` for each value of the parameter ``vary``.

    Parameters
    ----------
    problem : module
        A module of :mod:`cutfem.problems`.
    params : problem.Params
        Base parameters; ``vary`` is replaced by each value in turn.
    vary : str
        Name of the varied field of ``params``.
    values : iterable
    extras : callable, optional
        ``extras(system) -> dict[str, float]``: quantities of the assembled
        system, computed before the solve, e.g.
        ``lambda s: {"cond": core.condition_number(s.A)}``.

    Returns
    -------
    dict[str, np.ndarray]
        One array per column: ``vary``, ``h``, ``n_dofs``, the extras and the
        errors. A failed solve gives NaN errors.
    """
    rows, table = [], None
    for value in values:
        p = dataclasses.replace(params, **{vary: value})
        with problem.build(p) as system:
            row = {vary: value, "h": system.h, "n_dofs": system.n_dofs}
            if extras is not None:
                row.update(extras(system))
            try:
                row.update(problem.errors(system, problem.solve(system)))
            except RuntimeError as error:
                print0(f"  {vary} = {value}: {error}")
                row.update(dict.fromkeys(problem.ERRORS, np.nan))
        if table is None:
            table = Table(row)
            table.header()
        table.row(row)
        rows.append(row)
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


# Data files


def _to_json(obj):
    if dataclasses.is_dataclass(obj):
        return _to_json(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | np.ndarray):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _from_json(obj):
    if isinstance(obj, dict):
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and all(isinstance(v, int | float) for v in obj):
            return np.array(obj)
        return [_from_json(v) for v in obj]
    return obj


def _metadata(name):
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args],
                                capture_output=True, text=True)
        return result.stdout.strip()

    import dolfinx

    return {
        "study": name,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "mpi_ranks": MPI.COMM_WORLD.size,
        "dolfinx": dolfinx.__version__,
    }


def save_data(name, data):
    """Write ``data`` and metadata to ``results/<name>.json``."""
    path = RESULTS_DIR / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": _metadata(name), "data": _to_json(data)}
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_data(name):
    """Read the data saved by :func:`save_data`."""
    path = RESULTS_DIR / f"{name}.json"
    return _from_json(json.loads(path.read_text())["data"])


# Entry point


def run(name, compute, plot=None, arguments=None):
    """Compute (or reload) a study, save its data and draw its figures.

    Parameters
    ----------
    name : str
        Study name, e.g. ``"poisson/convergence"``; sets the output paths.
    compute : callable
        ``compute() -> data``; data is any nesting of dicts, lists, numbers,
        arrays and dataclasses.
    plot : callable, optional
        ``plot(data)`` returns a figure, or a dict ``{suffix: figure}`` for
        several figures (saved as ``<name>_<suffix>``).
    arguments : callable, optional
        ``arguments(parser)`` adds study-specific command line options; the
        parsed options are then passed as ``compute(args)``.
    """
    parser = argparse.ArgumentParser(description=f"Study {name}")
    if arguments is not None:
        arguments(parser)
    parser.add_argument("--plot-only", action="store_true",
                        help="redraw figures from saved data")
    parser.add_argument("--show", action="store_true",
                        help="show figures in a window")
    parser.add_argument("--formats", nargs="+", default=["pdf"],
                        help="figure formats (default: pdf)")
    args = parser.parse_args()

    if args.plot_only:
        data = load_data(name)
    else:
        data = compute(args) if arguments is not None else compute()
        if MPI.COMM_WORLD.rank == 0:
            print0(f"\nSaved {save_data(name, data).relative_to(ROOT)}")

    if plot is None or MPI.COMM_WORLD.rank != 0:
        return data

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cutfem import plotting

    plotting.use_style()
    figures = plot(_from_json(_to_json(data)))
    if not isinstance(figures, dict):
        figures = {"": figures}
    for suffix, fig in figures.items():
        stem = FIGURES_DIR / (f"{name}_{suffix}" if suffix else name)
        for path in plotting.save(fig, stem, args.formats):
            print0(f"Saved {path.relative_to(ROOT)}")

    if args.show:
        plt.show()
    return data
