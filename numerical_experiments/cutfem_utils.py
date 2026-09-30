"""
cutfem_utils.py
Shared utility functions for the CutFEM numerical experiments.
"""

import dolfinx.mesh
import numpy as np
import ufl


def ghost_facet_measure(unf_mesh, cut_cells):
    """
    Build a dS measure restricted to interior facets of cut elements.
    These are the facets over which the ghost penalty acts.
    Returns (dS_ghost, n_facet)
    """
    cut_cell_set = set(cut_cells.tolist())
    tdim = unf_mesh.topology.dim
    fdim = tdim - 1
    unf_mesh.topology.create_connectivity(fdim, tdim)
    f2c = unf_mesh.topology.connectivity(fdim, tdim)

    ghost_facet_ids = np.array(
        [f for f in range(f2c.num_nodes)
         if len(f2c.links(f)) == 2
         and cut_cell_set.intersection(f2c.links(f).tolist())],
         dtype=np.int32,
    )
    ghost_facet_tags = dolfinx.mesh.meshtags(
        unf_mesh, fdim, ghost_facet_ids,
        np.ones(len(ghost_facet_ids), dtype=np.int32),
    )
    dS_ghost = ufl.Measure("dS", domain=unf_mesh,
                           subdomain_data=ghost_facet_tags, subdomain_id=1)
    n_facet = ufl.FacetNormal(unf_mesh)
    return dS_ghost, n_facet
