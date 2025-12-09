from __future__ import annotations

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .project_manager import Project


ProgressCallback = Callable[[float, str], None]


@dataclass
class StemSeparationResult:
    stems: dict[str, Path]  # stem_name -> absolute path


def _parse_progress(line: str) -> float | None:
    m = re.search(r"(\d+)%", line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def separate_stems_for_project(
    project: Project,
    model: str = "htdemucs",
    quality: str = "balanced",
    progress_cb: ProgressCallback | None = None,
) -> StemSeparationResult:

    """
    Run Demucs stem separation for the project's audio file.

    - Uses `python -m demucs -n <model> -o <temp_out_root> <audio>`
    - Moves produced stems into project.folder / "stems" / <model>/
    - Updates project.stems_by_model[model] accordingly
    """
    audio_path = project.get_audio_path()
    if not audio_path or not audio_path.is_file():
        raise FileNotFoundError("Audio file not found for this project.")

    audio_path = audio_path.resolve()

    def report(p: float, msg: str):
        if progress_cb is not None:
            progress_cb(p, msg)

    report(0.0, "Starting stem separation...")

    # Final stems folder for this model
    stems_dir = project.folder / "stems" / model
    stems_dir.mkdir(parents=True, exist_ok=True)

    # Temporary Demucs output directory
    temp_out_root = project.folder / "stems_temp"
    if temp_out_root.exists():
        shutil.rmtree(temp_out_root)
    temp_out_root.mkdir(parents=True, exist_ok=True)

    # Quality presets -> Demucs CLI options
    extra_args: list[str] = []
    q = (quality or "balanced").lower()

    if q == "fast":
        # Moins d'overlap, une seule passe
        extra_args = ["--overlap", "0.1", "--shifts", "1"]
    elif q == "hq":
        # Plus d'overlap, plusieurs passes -> meilleure qualité, plus lent
        extra_args = ["--overlap", "0.5", "--shifts", "5"]
    else:
        # balanced (proche du comportement par défaut)
        extra_args = ["--overlap", "0.25", "--shifts", "1"]

    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        model,
        "-o",
        str(temp_out_root),
        *extra_args,
        str(audio_path),
    ]


    report(0.0, f"Running Demucs ({model})...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        pct = _parse_progress(line)
        if pct is not None:
            report(pct, line)
        # sinon on ignore silencieusement

    retcode = proc.wait()
    if retcode != 0:
        raise RuntimeError(f"Demucs failed with exit code {retcode}.")

    report(100.0, "Demucs finished, collecting stems...")

    # Demucs output: <temp_out_root>/<model>/<track_name>/*.wav
    model_dir = temp_out_root / model
    if model_dir.exists():
        track_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    else:
        track_dirs = []

    if track_dirs:
        track_dir = track_dirs[0]
    else:
        track_dir = temp_out_root

    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    stem_paths: dict[str, Path] = {}

    for stem_file in track_dir.iterdir():
        if not stem_file.is_file():
            continue
        if stem_file.suffix.lower() not in exts:
            continue

        name = stem_file.stem.lower()
        dst = stems_dir / stem_file.name
        # overwrite if it exists for this model
        if dst.exists():
            dst.unlink()
        shutil.move(stem_file, dst)
        stem_paths[name] = dst

    # Clean up temp directory
    if temp_out_root.exists():
        shutil.rmtree(temp_out_root)

    # Update project metadata: stems_by_model[model] = {stem_name: rel_path}
    rel_mapping = {
        stem_name: str(path.relative_to(project.folder))
        for stem_name, path in stem_paths.items()
    }
    project.set_stems_for_model(model, rel_mapping)

    report(100.0, "Stems collected.")

    return StemSeparationResult(stems=stem_paths)
