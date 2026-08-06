"""Plot adaptive all-atom initialization scaling results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    colors = {"pe": "tab:blue", "p3ht": "tab:orange", "pes": "tab:green"}
    for system in colors:
        selected = sorted(
            (row for row in rows if row["system"] == system),
            key=lambda row: row["actual_atoms"],
        )
        if not selected:
            continue
        atoms = [row["actual_atoms"] for row in selected]
        axes[0].plot(
            atoms,
            [row["dpd_steps"] for row in selected],
            "o-",
            color=colors[system],
            label=system.upper(),
        )
        axes[1].loglog(
            atoms,
            [row["dpd_s"] for row in selected],
            "o-",
            color=colors[system],
            label=system.upper(),
        )
    pe = {
        row["actual_atoms"]: row
        for row in rows
        if row["system"] == "pe"
    }
    atoms = np.asarray(sorted(pe), dtype=float)
    memory = np.asarray([pe[int(atom)]["peak_rss_gib"] for atom in atoms])
    slope, intercept = np.polyfit(atoms, memory, 1)
    projected_atoms = np.asarray([*atoms, 800_000.0])
    axes[2].plot(atoms, memory, "o", color="tab:blue", label="Measured")
    axes[2].plot(
        projected_atoms,
        slope * projected_atoms + intercept,
        "--",
        color="tab:gray",
        label=f"Linear projection: {slope * 800_000 + intercept:.1f} GiB",
    )
    axes[0].set_ylabel("Adaptive DPD steps")
    axes[0].set_xlabel("Atoms")
    axes[0].legend()
    axes[1].set_ylabel("DPD wall time (s)")
    axes[1].set_xlabel("Atoms")
    axes[2].set_ylabel("Peak host memory (GiB)")
    axes[2].set_xlabel("Atoms")
    axes[2].legend()
    figure.savefig(args.output, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
