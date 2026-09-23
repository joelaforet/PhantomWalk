"""Ion-aggregate structure of a carboxylate ionomer with monatomic counterions.

Aggregates are the connected components of a graph whose nodes are the
counterions and the carboxylate groups, with an edge between an ion and a
group when any of the group's two oxygens is within `cutoff` of the ion
(minimum image). An aggregate's size is its number of counterions.
"""

import numpy as np
from scipy.spatial import cKDTree


def ion_sites(compound, counterion="Na"):
    """Return (ion indices, carboxylate groups as (C, O, O) index triples)."""
    particles = list(compound.particles())
    index = {p: i for i, p in enumerate(particles)}
    ions = [
        i for i, p in enumerate(particles) if p.element.symbol == counterion
    ]
    groups = []
    for p in particles:
        if p.element.symbol != "C":
            continue
        oxygens = [n for n in p.direct_bonds() if n.element.symbol == "O"]
        if len(oxygens) == 2 and all(
            len(list(o.direct_bonds())) == 1 for o in oxygens
        ):
            groups.append((index[p], index[oxygens[0]], index[oxygens[1]]))
    return np.asarray(ions, dtype=int), np.asarray(groups, dtype=int)


def ion_structure(positions_a, box_a, ions, groups, cutoff=3.0):
    """Coordination and aggregate statistics for one configuration.

    Parameters
    ----------
    positions_a : (N, 3) array, Angstrom (wrapped or unwrapped)
    box_a : (3,) orthorhombic box lengths, Angstrom
    ions : (n_ion,) indices of the counterions
    groups : (n_group, 3) indices (C, O, O) of the carboxylates
    cutoff : ion-oxygen distance defining contact, Angstrom

    """
    box = np.asarray(box_a, dtype=float)
    wrapped = np.mod(np.asarray(positions_a, dtype=float), box)
    oxygens = groups[:, 1:].ravel()
    owner = np.repeat(np.arange(len(groups)), 2)
    tree = cKDTree(wrapped[oxygens], boxsize=box)
    ion_xyz = wrapped[ions]
    contacts = tree.query_ball_point(ion_xyz, r=cutoff)
    nearest, _ = tree.query(ion_xyz, k=1)

    n_ion, n_group = len(ions), len(groups)
    parent = list(range(n_ion + n_group))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    coordination = np.zeros(n_ion, dtype=int)
    for i, hits in enumerate(contacts):
        coordination[i] = len(hits)
        for h in hits:
            ra, rb = find(i), find(n_ion + owner[h])
            if ra != rb:
                parent[ra] = rb
    sizes = {}
    for i in range(n_ion):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    sizes = np.asarray(sorted(sizes.values(), reverse=True), dtype=int)
    histogram = np.bincount(sizes)
    ion_weighted = histogram * np.arange(len(histogram))
    return {
        "cutoff_a": cutoff,
        "n_ions": int(n_ion),
        "n_carboxylates": int(n_group),
        "free_ion_fraction": float(np.mean(coordination == 0)),
        "mean_oxygen_coordination": float(coordination.mean()),
        "coordination_histogram": np.bincount(coordination).tolist(),
        "nearest_oxygen_a_percentiles": {
            str(q): float(np.percentile(nearest, q))
            for q in (5, 25, 50, 75, 95)
        },
        "n_aggregates": int(len(sizes)),
        "largest_aggregate": int(sizes.max()) if len(sizes) else 0,
        "number_mean_size": float(sizes.mean()) if len(sizes) else 0.0,
        "ion_weighted_mean_size": float((sizes**2).sum() / sizes.sum())
        if len(sizes)
        else 0.0,
        # aggregate_size_histogram[k] = number of aggregates with k ions;
        # ion_fraction_by_size[k] = fraction of all ions in such aggregates
        "aggregate_size_histogram": histogram.tolist(),
        "ion_fraction_by_size": (ion_weighted / max(n_ion, 1)).tolist(),
    }
