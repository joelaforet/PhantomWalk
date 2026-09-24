"""All-atom PhantomWalk: hand-off and analysis around FlowerMD's initializer.

FlowerMD (``flowermd.library.AllAtomPhantomWalk`` and friends) builds and
relaxes the melt. This package holds what the paper's experiments need
around it: the Sage 2.3.0 hand-off minimization with its energy-removed
metric, structural analysis such as ionomer ion aggregates, and
residue-aware PDB/trajectory export with NGLView and PyMOL views
(``structure`` and ``visualization``; the latter is imported on demand).
"""

from .handoff import build_sage_interchange, minimize_by_component, sage_handoff
from .ion_clusters import ion_sites, ion_structure
from .structure import (
    Topology,
    flush_trajectory,
    frame_positions,
    residue_topology,
    trajectory_positions,
    universe,
    write_pdb,
    write_trajectory,
)

__all__ = [
    "Topology",
    "build_sage_interchange",
    "flush_trajectory",
    "frame_positions",
    "ion_sites",
    "ion_structure",
    "minimize_by_component",
    "residue_topology",
    "sage_handoff",
    "trajectory_positions",
    "universe",
    "write_pdb",
    "write_trajectory",
]
