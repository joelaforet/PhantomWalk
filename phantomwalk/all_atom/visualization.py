"""NGLView and PyMOL views of exported all-atom melts.

Structures are the PDB files written by `phantomwalk.all_atom.structure`:
one residue per monomer, one segment per molecule, a blank chain ID, and
CONECT records for every bond. Everything is drawn as licorice.

NGL and PyMOL both add distance-based bonds between atoms of non-standard
residues, which is every residue of a synthetic polymer. During DPD atoms
overlap, so those guesses would draw bonds that do not exist. The NGL views
here keep only the CONECT bonds, and the PyMOL script sets
``connect_mode 1``, which reads CONECT only.

The in-memory trajectory adaptor, the player widget and `save_gif` follow
``demo_utils.py`` of joelaforet/mbuild_protein_demos.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

# Colors of NGL's "chainname" scheme follow the chain name, which NGL takes
# from the segment ID when the chain ID column is blank: one color per
# molecule.
COLOR_SCHEMES = ("chainname", "resname", "element")
PYMOL_PALETTE = (
    "marine", "orange", "forest", "firebrick", "purpleblue", "olive",
    "deepteal", "salmon", "violetpurple", "sand", "slate", "chocolate",
    "limon", "hotpink", "teal", "wheat", "deepsalmon", "lightblue",
)


def _pdb_text(source):
    text = str(source)
    if "\n" not in text:
        return Path(text).read_text()
    return text


def _conect_pairs(text):
    pairs = []
    for line in text.splitlines():
        if line.startswith("CONECT"):
            serials = [
                int(line[i : i + 5]) for i in range(6, len(line.rstrip()), 5)
            ]
            pairs.extend(
                min(serials[0], other) * 100_000 + max(serials[0], other)
                for other in serials[1:]
            )
    return sorted(set(pairs))


def _explicit_bonds_only(view, text):
    """Replace NGL's bond graph with the CONECT bonds of `text`."""
    pairs = _conect_pairs(text)
    if not pairs:
        return
    # Runs in the browser after the component loads, before styling. Keys
    # are min(serial) * 100000 + max(serial); PDB serials stay below 1e5.
    view._execute_js_code(
        """
        const structure = this.stage.compList[0].structure;
        const explicit = new Set(PAIRS);
        const store = structure.bondStore;
        const a = structure.getAtomProxy(), b = structure.getAtomProxy();
        let count = 0;
        for (let i = 0; i < store.count; i++) {
            a.index = store.atomIndex1[i]; b.index = store.atomIndex2[i];
            const key = Math.min(a.serial, b.serial) * 100000
                + Math.max(a.serial, b.serial);
            if (!explicit.has(key)) continue;
            explicit.delete(key);
            store.atomIndex1[count] = a.index;
            store.atomIndex2[count] = b.index;
            store.bondOrder[count] = store.bondOrder[i];
            count++;
        }
        store.count = count;
        structure.finalizeBonds();
        structure.refreshPosition();
        """.replace("PAIRS", json.dumps(pairs))
    )


def _style(view, color, radius, unitcell, spheres):
    if color not in COLOR_SCHEMES:
        raise ValueError(f"color must be one of {COLOR_SCHEMES}.")
    view.clear_representations()
    # "licorice" is NGL's stick representation. For "element", carbons are
    # grey; the other schemes color whole residues or molecules.
    view.add_representation("licorice", selection="all", color=color,
                            radius=radius)
    if spheres:
        view.add_representation("spacefill", selection=spheres, color=color,
                                radiusScale=0.5)
    if unitcell:
        view.add_representation("unitcell")
    view.center()


def _text_frames_class():
    from nglview.base_adaptor import Structure, Trajectory

    class TextFrames(Structure, Trajectory):
        """One PDB text plus an in-memory (n_frames, N, 3) Angstrom array."""

        def __init__(self, text, frames):
            Structure.__init__(self)
            Trajectory.__init__(self)
            self.ext = "pdb"
            self.params = {}
            self._text = text
            self._frames = np.asarray(frames, dtype=np.float32)

        def get_structure_string(self):
            return self._text

        def get_coordinates(self, index):
            return self._frames[index]

        @property
        def n_frames(self):
            return len(self._frames)

    return TextFrames


def show_structure(
    source,
    color="chainname",
    radius=0.15,
    unitcell=True,
    spheres=None,
    width="700px",
    height="500px",
):
    """Show an exported melt as licorice in NGLView.

    Parameters
    ----------
    source : str or path-like
        A PDB path from `structure.write_pdb`, or PDB text.
    color : {"chainname", "resname", "element"}, default "chainname"
        One color per molecule, per residue name (e.g. PSR against PSS for
        tacticity, ETH/ACR/NA for the ionomer), or per element.
    radius : float, default 0.15
        Stick radius in Angstrom.
    unitcell : bool, default True
        Draw the periodic box from the CRYST1 record.
    spheres : str, optional
        NGL selection also drawn as spheres, for atoms without bonds that
        licorice leaves nearly invisible, e.g. ``"[NA]"`` for Na+ residues.
    width, height : str
        Widget size as CSS lengths.
    """
    import nglview

    text = _pdb_text(source)
    view = nglview.NGLWidget(nglview.TextStructure(text, ext="pdb"),
                             default_representation=False)
    _explicit_bonds_only(view, text)
    _style(view, color, radius, unitcell, spheres)
    view.layout.width = width
    view.layout.height = height
    return view


def _movie_view(ngl_widget):
    import ipywidgets as widgets

    play = widgets.Play(min=0, max=ngl_widget.max_frame, interval=100)
    slider = widgets.IntSlider(
        min=0, max=ngl_widget.max_frame, description="Frame",
        layout=widgets.Layout(width="400px"),
    )
    box = widgets.VBox([ngl_widget, widgets.HBox([play, slider])])
    # jslink runs playback in the browser; keep the links alive on the box.
    box._links = [
        widgets.jslink((play, "value"), (slider, "value")),
        widgets.jslink((slider, "value"), (ngl_widget, "frame")),
    ]
    box.ngl_widget = ngl_widget
    return box


def show_movie(
    pdb,
    trajectory,
    color="chainname",
    radius=0.15,
    unitcell=True,
    spheres=None,
    width="700px",
    height="500px",
):
    """Play a trajectory in NGLView with a play button and frame slider.

    Parameters
    ----------
    pdb : str or path-like
        Topology PDB from `structure.write_pdb` (same atom order).
    trajectory : str, path-like or array
        A DCD/XTC from `structure.write_trajectory`, or frames of shape
        (n_frames, N, 3) in Angstrom from `structure.trajectory_positions`.
    color, radius, unitcell, spheres, width, height
        As in `show_structure`.

    Returns
    -------
    ipywidgets.VBox
        The viewer and its controls; ``.ngl_widget`` is the NGL widget, for
        `save_gif`.
    """
    import nglview

    text = _pdb_text(pdb)
    if isinstance(trajectory, (str, os.PathLike)):
        import MDAnalysis as mda

        u = mda.Universe(str(pdb), str(trajectory))
        frames = np.asarray([u.atoms.positions.copy() for _ in u.trajectory])
    else:
        frames = np.asarray(trajectory)
    if frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError(
            f"frames must be (n_frames, N, 3), got {frames.shape}."
        )
    view = nglview.NGLWidget(_text_frames_class()(text, frames),
                             default_representation=False)
    _explicit_bonds_only(view, text)
    _style(view, color, radius, unitcell, spheres)
    view.layout.width = width
    view.layout.height = height
    return _movie_view(view)


def save_gif(view, n_frames, path, duration=120, loop=0, timeout=30.0):
    """Write the frames of a displayed movie widget to an animated GIF.

    ``NGLWidget.render_image`` asks the browser for each picture, so the
    widget must be displayed in a live notebook first; under nbconvert
    there is no front end and this raises.

    Parameters
    ----------
    view : widget from `show_movie`, or an nglview.NGLWidget
    n_frames : int
    path : str or path-like
    duration : int, default 120
        Milliseconds per frame.
    loop : int, default 0
        GIF loop count; 0 repeats forever.
    timeout : float, default 30
        Seconds to wait for each picture.
    """
    from io import BytesIO

    from PIL import Image

    view = getattr(view, "ngl_widget", view)
    images = []
    for index in range(n_frames):
        view.frame = index
        image = view.render_image()
        deadline = time.monotonic() + timeout
        while not image.value and time.monotonic() < deadline:
            time.sleep(0.1)
        if not image.value:
            raise RuntimeError(
                f"No picture for frame {index} within {timeout} s. Display "
                "the widget in a live notebook before calling save_gif."
            )
        images.append(Image.open(BytesIO(image.value)).convert("P"))
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=duration, loop=loop)
    return path


def write_pymol_script(pdb, path, trajectory=None, color_by="segi",
                       stick_radius=0.15, fps=15):
    """Write a PyMOL script that loads a structure or movie as licorice.

    Carbons take one color per molecule (``color_by="segi"``) or per
    residue name (``"resn"``); other elements keep their element colors.
    The script only sets up the scene and the movie; the rendering commands
    are at its end, commented, to adjust and run from PyMOL.

    File paths in the script are relative to the script's folder, so run
    PyMOL from there: ``cd <folder> && pymol movie.pml``.

    Parameters
    ----------
    pdb : str or path-like
        Topology PDB from `structure.write_pdb`.
    path : str or path-like
        The ``.pml`` file to write.
    trajectory : str or path-like, optional
        A DCD from `structure.write_trajectory`; each frame becomes a state.
    color_by : {"segi", "resn"}, default "segi"
    stick_radius : float, default 0.15
    fps : int, default 15
        Movie playback rate.
    """
    if color_by not in ("segi", "resn"):
        raise ValueError('color_by must be "segi" or "resn".')
    path = Path(path)
    folder = path.parent.resolve()

    def rel(p):
        return os.path.relpath(Path(p).resolve(), folder)

    lines = [
        "# PyMOL scene for a PhantomWalk all-atom melt.",
        f"# Run from this folder: pymol {path.name}",
        "reinitialize",
        "# CONECT bonds only; PyMOL's distance-based guesses would bond",
        "# atoms that overlap during DPD.",
        "set connect_mode, 1",
        f"load {rel(pdb)}, melt",
    ]
    if trajectory is not None:
        lines += [
            f"load_traj {rel(trajectory)}, melt, state=1",
            "# Optional: average out thermal jitter for a smoother movie.",
            "# smooth melt, 30, 3",
        ]
    lines += [
        "hide everything",
        "show sticks",
        "# Atoms without bonds (counterions) as spheres.",
        "show spheres, melt and not (bound_to melt)",
        "set sphere_scale, 0.5",
        f"set stick_radius, {stick_radius}",
        "set valence, 0",
        "bg_color white",
        "set ray_opaque_background, 1",
        "set specular, 0.25",
        "set ambient, 0.35",
        "set cell_color, grey60",
        "show cell",
        "python",
        f"palette = {list(PYMOL_PALETTE)!r}",
        "keys = set()",
        f'cmd.iterate("melt", "keys.add({color_by})", space={{"keys": keys}})',
        "for i, key in enumerate(sorted(keys)):",
        f'    cmd.color(palette[i % len(palette)], f"melt and {color_by} {{key}} and elem C")',
        "python end",
        'util.cnc("melt")',
        "color grey90, melt and elem H",
        "orient melt",
    ]
    if trajectory is not None:
        lines += [
            f"set movie_fps, {fps}",
            "# One movie frame per trajectory state.",
            'python\ncmd.mset(f"1 -{cmd.count_states(\'melt\')}")\npython end',
            "frame 1",
        ]
    lines += [
        "",
        "# Rendering (uncomment and adjust):",
        "# set ray_trace_frames, 1",
        "# viewport 1600, 1200",
        "# mpng frames/frame_          (movie: one PNG per state)",
        "# png melt.png, width=2400, height=1800, dpi=300, ray=1",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path
