"""Step 0: sanity check of the cut gyroid geometry.

For increasing resolution (background cells per unit-cell length):

* fluid volume, which by symmetry is exactly half the box volume;
* immersed surface area per unit-cell volume (Schoen gyroid: ~3.09 / L);
* fluid area of the inlet face x = 0 (half, by symmetry) and the moment
  int y^2 z over it by cut facet quadrature, against a fine raster of the
  analytic level set (check of the exterior-facet cut path);
* smallest cut-cell volume fraction |T cap Omega| / |T|;
* active / cut cells and the mesh and quadrature creation times.

Then a PyVista view of the fluid domain and of the gyroid surface.
"""

import dataclasses

import numpy as np

from cutfem import study
from cutfem.problems import gyroid

N_CELL = [4, 8, 12, 16]
PARAMS = gyroid.Params()
FIGURE = study.FIGURES_DIR / "gyroid" / "geometry.png"


def inlet_raster(params, n=4000):
    """Fluid area of x = 0 and moment int y^2 z from the sign of phi(0, y, z)
    on an n x n raster (midpoint rule; error ~1/n from the cut pixels)."""
    t = (np.arange(n) + 0.5) / n * params.L
    y, z = np.meshgrid(t, t, indexing="ij")
    w = 2 * np.pi / params.L
    fluid = np.sin(w * y) * np.cos(w * z) + np.sin(w * z) <= 0.0   # x = 0
    pixel = (params.L / n) ** 2
    return float(fluid.sum() * pixel), float((y ** 2 * z)[fluid].sum() * pixel)


def compute():
    """Geometry quantities for every resolution."""
    study.print0(PARAMS)
    area_ref, moment_ref = inlet_raster(PARAMS)
    study.print0(f"inlet face, raster 4000^2: fluid area {area_ref:.6f}, "
                 f"int y^2 z {moment_ref:.6f} (full face {PARAMS.L ** 5 / 6:.6f})")
    rows, table = [], None
    for n_cell in N_CELL:
        row = gyroid.geometry(dataclasses.replace(PARAMS, n_cell=n_cell))
        row["inlet_moment_error"] = abs(row["inlet_moment"] - moment_ref) / moment_ref
        if table is None:
            table = study.Table(row)
            table.header()
        table.row(row)
        rows.append(row)
    return {"params": PARAMS, "inlet_area_raster": area_ref,
            "inlet_moment_raster": moment_ref,
            "columns": {k: np.array([r[k] for r in rows]) for k in rows[0]}}


def show(params, path, degree=3):
    """Fluid domain (left) and gyroid surface (right), saved as a screenshot."""
    import pyvista as pv
    import qugar.plot
    import qugar.reparam

    mesh, _, _ = gyroid.gyroid_mesh(params)
    solid = qugar.plot.reparam_mesh_to_PyVista(
        qugar.reparam.create_reparam_mesh(mesh, degree=degree, levelset=False))
    surface = qugar.plot.reparam_mesh_to_PyVista(
        qugar.reparam.create_reparam_mesh(mesh, degree=degree, levelset=True))

    plotter = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(1800, 700))
    plotter.subplot(0, 0)
    plotter.add_text("fluid domain", font_size=11)
    plotter.add_mesh(solid.get("reparam"), color="lightsteelblue")
    plotter.add_mesh(pv.Box(bounds=(0, params.N * params.L, 0, params.L, 0, params.L)),
                     style="wireframe", color="black")
    plotter.view_isometric()
    plotter.subplot(0, 1)
    plotter.add_text("gyroid surface", font_size=11)
    plotter.add_mesh(surface.get("reparam"), color="orange", smooth_shading=True)
    plotter.add_mesh(pv.Box(bounds=(0, params.N * params.L, 0, params.L, 0, params.L)),
                     style="wireframe", color="black")
    plotter.view_isometric()
    plotter.link_views()
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(path))
    plotter.close()
    study.print0(f"Saved {path.relative_to(study.ROOT)}")


if __name__ == "__main__":
    study.run("gyroid/geometry", compute)
    show(dataclasses.replace(PARAMS, n_cell=8), FIGURE)
