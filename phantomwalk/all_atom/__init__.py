"""All-atom PhantomWalk: hand-off and analysis around FlowerMD's initializer.

FlowerMD (``flowermd.library.AllAtomPhantomWalk`` and friends) builds and
relaxes the melt. This package holds what the paper's experiments need
around it: the Sage 2.3.0 hand-off minimization with its energy-removed
metric, and structural analysis such as ionomer ion aggregates.
"""

from .handoff import build_sage_interchange, minimize_by_component, sage_handoff
from .ion_clusters import ion_sites, ion_structure

__all__ = [
    "build_sage_interchange",
    "ion_sites",
    "ion_structure",
    "minimize_by_component",
    "sage_handoff",
]
