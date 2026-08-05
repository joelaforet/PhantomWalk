"""Plot the dense all-atom DPD/FIRE iteration sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def plot_grid(rows: list[dict], output: Path) -> None:
    """Plot pass/fail state and FastFIRE wall time for the screening grid."""

    systems = ("pe", "p3ht", "pes")
    dpd_steps = sorted({row["dpd_steps"] for row in rows})
    fire_steps = sorted({row["fire_steps"] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, system in zip(axes, systems, strict=True):
        selected = {(
            row["dpd_steps"], row["fire_steps"]
        ): row for row in rows if row["system"] == system}
        values = np.array([
            [selected[(dpd, fire)]["fastfire_s"] for fire in fire_steps]
            for dpd in dpd_steps
        ])
        image = axis.imshow(values, cmap="viridis", aspect="auto")
        for y, dpd in enumerate(dpd_steps):
            for x, fire in enumerate(fire_steps):
                row = selected[(dpd, fire)]
                mark = "✓" if row["sensible"] else "×"
                axis.text(x, y, f"{values[y, x]:.1f}\n{mark}", ha="center", va="center")
        axis.set_title(system.upper())
        axis.set_xticks(range(len(fire_steps)), fire_steps)
        axis.set_yticks(range(len(dpd_steps)), dpd_steps)
        axis.set_xlabel("FIRE steps")
        axis.set_ylabel("DPD steps")
    figure.colorbar(image, ax=axes, label="FastFIRE wall time (s)")
    figure.savefig(output, dpi=200)
    plt.close(figure)


def plot_finalists(rows: list[dict], output: Path) -> None:
    """Compare the two finalists at approximately 10,000 atoms."""

    protocols = ((250, 400), (1000, 50), (1000, 100))
    systems = ("pe", "p3ht", "pes")
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["dpd_steps"], row["fire_steps"])][row["system"]] = row
    x = np.arange(len(systems))
    width = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for index, protocol in enumerate(protocols):
        selected = [grouped[protocol][system] for system in systems]
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            [row["fastfire_s"] for row in selected],
            width,
            label=f"{protocol[0]}/{protocol[1]}",
        )
        axes[1].bar(
            x + offset,
            [row["openmm_initial_energy_per_atom"] for row in selected],
            width,
        )
    axes[0].set_ylabel("FastFIRE wall time (s)")
    axes[0].legend(title="DPD/FIRE")
    axes[1].set_ylabel("Initial OpenMM energy (kJ mol⁻¹ atom⁻¹)")
    axes[1].set_yscale("log")
    for axis in axes:
        axis.set_xticks(x, [system.upper() for system in systems])
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screen", type=Path)
    parser.add_argument("finalists", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_grid(_read(args.screen), args.output / "screening_grid.png")
    plot_finalists(_read(args.finalists), args.output / "finalists_10k.png")


if __name__ == "__main__":
    main()
