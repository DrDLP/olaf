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

    Progress behavior:
    - Demucs tqdm progress (possibly multiple bars) is mapped into 0..90%.
    - Post-processing (moving files + saving metadata) is mapped into 90..100%.
    - Progress is monotonic (never goes backwards) and never hits 100% too early.
    """
    from collections import deque
    import os

    audio_path = project.get_audio_path()
    if not audio_path or not audio_path.is_file():
        raise FileNotFoundError("Audio file not found for this project.")
    audio_path = audio_path.resolve()

    def report(p: float, msg: str):
        if progress_cb is not None:
            progress_cb(p, msg)

    report(0.0, "Starting stem separation...")

    stems_dir = project.folder / "stems" / model
    stems_dir.mkdir(parents=True, exist_ok=True)

    temp_out_root = project.folder / "stems_temp"
    if temp_out_root.exists():
        shutil.rmtree(temp_out_root)
    temp_out_root.mkdir(parents=True, exist_ok=True)

    # Quality presets -> Demucs CLI options
    extra_args: list[str] = []
    q = (quality or "balanced").lower()
    if q == "fast":
        extra_args = ["--overlap", "0.1", "--shifts", "1"]
    elif q == "hq":
        extra_args = ["--overlap", "0.5", "--shifts", "5"]
    else:
        extra_args = ["--overlap", "0.25", "--shifts", "1"]

    # -----------------------
    # Global progress mapping (Demucs -> 0..90)
    # -----------------------
    overall_pct = 0.0        # monotonic overall progress (0..100)
    demucs_cap = 90.0        # never go above this while Demucs runs
    demucs_bar_index = 0
    demucs_last_pct: float | None = None

    def _set_overall(p: float, msg: str):
        nonlocal overall_pct
        p = max(0.0, min(100.0, float(p)))
        overall_pct = max(overall_pct, p)
        report(overall_pct, msg)

    def _map_demucs_progress(local_pct: float) -> float:
        """
        Map Demucs local percent (0..100) into overall (0..demucs_cap).
        Demucs can restart tqdm bars; we detect big drops and treat them as a new bar.
        """
        nonlocal demucs_bar_index, demucs_last_pct

        lp = max(0.0, min(100.0, float(local_pct)))

        # Detect bar reset (e.g., 99 -> 1)
        if demucs_last_pct is not None and lp + 5.0 < demucs_last_pct:
            demucs_bar_index += 1
            demucs_last_pct = lp
        else:
            demucs_last_pct = lp if demucs_last_pct is None else max(demucs_last_pct, lp)

        # Most of the range is bar 0; later bars share the last small slice.
        bar0_end = 85.0
        if demucs_bar_index <= 0:
            mapped = (lp / 100.0) * bar0_end
        else:
            mapped = bar0_end + (lp / 100.0) * (demucs_cap - bar0_end)

        return min(mapped, demucs_cap)

    def run_demucs(device: str | None) -> tuple[int, list[str], list[str]]:
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            model,
            "-o",
            str(temp_out_root),
            *extra_args,
        ]
        if device:
            cmd += ["-d", device]
        cmd.append(str(audio_path))

        _set_overall(overall_pct, f"Running Demucs ({model})" + (f" on {device.upper()}..." if device else "..."))

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        full: list[str] = []
        tail = deque(maxlen=200)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            full.append(line)
            tail.append(line)

            pct = _parse_progress(line)
            if pct is not None:
                _set_overall(_map_demucs_progress(pct), line)
            else:
                # Forward only useful status lines, without messing with progress.
                if any(k in line.lower() for k in ("error", "ffmpeg", "cuda", "traceback", "exception", "no module", "saving", "export")):
                    _set_overall(overall_pct, line)

        retcode = proc.wait()
        return retcode, full, list(tail)

    # First try
    retcode, full_lines, tail_lines = run_demucs(device=None)

    # Retry on CPU if CUDA-like error
    if retcode != 0:
        joined = "\n".join(tail_lines).lower()
        cuda_like = any(
            s in joined
            for s in (
                "cuda out of memory",
                "cuda error",
                "cudnn",
                "device-side assert",
                "illegal memory access",
            )
        )
        if cuda_like:
            _set_overall(overall_pct, "Demucs failed on GPU, retrying on CPU...")

            if temp_out_root.exists():
                shutil.rmtree(temp_out_root)
            temp_out_root.mkdir(parents=True, exist_ok=True)

            # IMPORTANT: reset the Demucs bar detection (NO 'nonlocal' here!)
            demucs_bar_index = 0
            demucs_last_pct = None

            retcode, full_lines, tail_lines = run_demucs(device="cpu")

    if retcode != 0:
        tail = "\n".join(tail_lines).strip()
        raise RuntimeError(
            f"Demucs failed with exit code {retcode}.\n\nLast output lines:\n{tail}"
        )

    # -----------------------
    # Post-processing (90..100)
    # -----------------------
    _set_overall(max(overall_pct, demucs_cap), "Demucs finished, collecting stems...")

    model_dir = temp_out_root / model
    track_dirs = [p for p in model_dir.iterdir() if p.is_dir()] if model_dir.exists() else []
    track_dir = track_dirs[0] if track_dirs else temp_out_root

    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    files_to_move = [p for p in track_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    stem_paths: dict[str, Path] = {}
    move_start = max(overall_pct, 90.0)
    move_end = 99.0
    span = max(0.0, move_end - move_start)
    total = max(1, len(files_to_move))

    for i, stem_file in enumerate(files_to_move, start=1):
        name = stem_file.stem.lower()
        dst = stems_dir / stem_file.name
        if dst.exists():
            dst.unlink()
        shutil.move(stem_file, dst)
        stem_paths[name] = dst

        _set_overall(move_start + (i / total) * span, f"Collecting stems: {stem_file.name}")

    if temp_out_root.exists():
        shutil.rmtree(temp_out_root)

    rel_mapping = {stem_name: str(path.relative_to(project.folder)) for stem_name, path in stem_paths.items()}
    project.set_stems_for_model(model, rel_mapping)

    _set_overall(100.0, "Stems collected.")
    return StemSeparationResult(stems=stem_paths)
