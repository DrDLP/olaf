from __future__ import annotations

import json
import uuid
import shutil
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Root folder for all projects
PROJECTS_ROOT = Path("projects")

# Global metadata (e.g. next project number)
GLOBAL_META_PATH = PROJECTS_ROOT / "_projects_meta.json"


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _init_storage() -> None:
    """Ensure the projects root folder exists on disk."""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


def _load_global_meta() -> Dict[str, Any]:
    """
    Load global metadata (like 'next_project_number').

    Stored in projects/_projects_meta.json so it survives between runs.
    """
    _init_storage()
    if GLOBAL_META_PATH.is_file():
        try:
            with GLOBAL_META_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_global_meta(meta: Dict[str, Any]) -> None:
    """Persist global metadata."""
    _init_storage()
    with GLOBAL_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _sanitize_folder_piece(name: str) -> str:
    """
    Sanitize text for use in a folder name.

    Removes characters invalid on Windows ('<>:\"/\\|?*') and collapses
    whitespace. Keeps things readable but safe.
    """
    invalid = '<>:"/\\|?*'
    cleaned = "".join(c for c in name if c not in invalid)
    cleaned = " ".join(cleaned.split()).strip()  # collapse spaces

    if not cleaned:
        cleaned = "Untitled"

    # Avoid extremely long folder names
    if len(cleaned) > 60:
        cleaned = cleaned[:57].rstrip() + "..."
    return cleaned


def _build_folder_name(project_number: int, project_name: str) -> str:
    """Build folder name: 'Project #n (Name)'."""
    safe_name = _sanitize_folder_piece(project_name)
    return f"Project #{project_number} ({safe_name})"


# ----------------------------------------------------------------------
# Project dataclass
# ----------------------------------------------------------------------


@dataclass
class Project:
    """
    Represents a single Olaf project.

    NOTE: The folder name on disk is no longer the 'id'.
    New projects use 'folder_name' = 'Project #n (Name)'.
    Older projects without folder_name still work: we fall back to 'id'.
    """

    # Stable internal ID (UUID string)
    id: str
    name: str
    created_at: str
    updated_at: str

    # Human readable numbering / folder
    project_number: Optional[int] = None
    folder_name: Optional[str] = None

    # Files (paths stored relative to project.folder)
    audio_file: Optional[str] = None
    cover_file: Optional[str] = None
    lyrics_text: str = ""

    # Optional audio analysis
    audio_bpm: Optional[float] = None
    audio_duration_sec: Optional[float] = None
    audio_waveform_image: Optional[str] = None  # relative path

    # Stems indexed by model name
    #   stems_by_model[model_name][stem_name] = relative path
    stems_by_model: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # --- Visualization state (3D visualizations tab) -------------------
    # Legacy "single plugin" fields (kept for compatibility)
    visualization_plugin_id: Optional[str] = None
    visualization_parameters: Dict[str, Any] = field(default_factory=dict)
    visualization_routing: Dict[str, str] = field(default_factory=dict)
    
    # --- NEW: 2D cover visualizations ---------------------------------
    # Per-effect configuration:
    #   cover_visual_effects[effect_id] = {
    #       "parameters": {...},
    #       "routing": {"audio_source": "main" or "stem:drums", ...}
    #   }
    cover_visual_effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Ordered list of effect_ids forming the 2D chain
    cover_visual_chain: List[str] = field(default_factory=list) 

    # --- Lyrics visualizations (text-based) ---------------------------
    # Global lyrics visualization style for this project (karaoke, scrolling, etc.).
    lyrics_visual_plugin_id: Optional[str] = None
    lyrics_visual_parameters: Dict[str, Any] = field(default_factory=dict)
    lyrics_visual_routing: Dict[str, str] = field(default_factory=dict)

    # Per-lyrics-plugin state, keyed by plugin_id.
    # This lets the user keep independent settings when switching between
    # different lyrics visualizations within the same project.
    lyrics_visual_parameters_by_plugin: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lyrics_visual_routing_by_plugin: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # New: per-plugin state, keyed by plugin_id, for the main audio visualizations
    #   visualizations[plugin_id] = {
    #       "routing": {...},
    #       "parameters": {...},
    #   }
    visualizations: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    # ------------------------------------------------------------------ #
    # Paths and persistence
    # ------------------------------------------------------------------ #

    @property
    def folder(self) -> Path:
        """
        Folder on disk where this project lives.

        - For new projects: use 'folder_name' (Project #n (Name)).
        - For old projects: fall back to 'id' (previous behaviour).
        """
        folder_id = self.folder_name or self.id
        return PROJECTS_ROOT / folder_id

    @property
    def metadata_path(self) -> Path:
        return self.folder / "project.json"

    def save(self) -> None:
        """Persist project metadata to project.json."""
        _init_storage()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.utcnow().isoformat()
        with self.metadata_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Stems management
    # ------------------------------------------------------------------ #

    def clear_all_stems(self) -> None:
        """Remove all stem references (does not delete audio files)."""
        self.stems_by_model.clear()
        self.save()

    # Backward compatible alias (older code may call clear_stems()).
    def clear_stems(self) -> None:
        self.clear_all_stems()

    def get_stems_for_model(self, model: str) -> Dict[str, str]:
        """Return mapping stem_name -> relative path for a given model."""
        return self.stems_by_model.get(model, {})

    def set_stems_for_model(self, model: str, stems: Dict[str, str]) -> None:
        """Set mapping stem_name -> relative path for a given model."""
        self.stems_by_model[model] = stems
        self.save()

    # ------------------------------------------------------------------ #
    # File management
    # ------------------------------------------------------------------ #

    def clear_analysis(self) -> None:
        """Clear audio analysis fields."""
        self.audio_bpm = None
        self.audio_duration_sec = None
        self.audio_waveform_image = None
        self.save()

    def set_audio_from_path(self, source: Path) -> None:
        """
        Copy the given audio file into the project and update audio_file.

        Stored as: audio/<original_name>
        """
        _init_storage()
        self.folder.mkdir(parents=True, exist_ok=True)
        audio_dir = self.folder / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        dest = audio_dir / source.name
        shutil.copy2(source, dest)

        self.audio_file = str(dest.relative_to(self.folder))
        self.save()

    def set_cover_from_path(self, source: Path) -> None:
        """
        Copy the given cover image into the project and update cover_file.

        Stored as: cover/<original_name>
        """
        _init_storage()
        self.folder.mkdir(parents=True, exist_ok=True)
        cover_dir = self.folder / "cover"
        cover_dir.mkdir(parents=True, exist_ok=True)

        dest = cover_dir / source.name
        shutil.copy2(source, dest)

        self.cover_file = str(dest.relative_to(self.folder))
        self.save()

    def get_audio_path(self) -> Optional[Path]:
        """Return absolute path to main audio file, if any."""
        if not self.audio_file:
            return None
        return self.folder / self.audio_file
        
    def get_preview_mix_path(self) -> Path:
        """Return absolute path to preview_mix.wav (may not exist yet)."""
        return self.folder / "preview_mix.wav"

    def ensure_preview_mix(self, force: bool = False) -> Optional[Path]:
        """
        Ensure 'preview_mix.wav' exists in the project root.

        This provides a stable WAV source for preview/playback and downstream
        processing. If the main audio is already a WAV, we copy it. Otherwise
        we attempt a conversion via ffmpeg.

        Args:
            force: If True, rebuild even if preview_mix.wav already exists.

        Returns:
            Absolute path to preview_mix.wav, or None if the project has no audio yet.

        Raises:
            RuntimeError: if ffmpeg is missing or conversion fails.
        """
        src = self.get_audio_path()
        if src is None or not src.is_file():
            return None

        out_path = self.get_preview_mix_path()

        # Skip rebuild if preview is newer than (or equal to) the source.
        if not force and out_path.is_file():
            try:
                if out_path.stat().st_mtime >= src.stat().st_mtime:
                    return out_path
            except Exception:
                return out_path

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Fast path: WAV -> WAV copy
        if src.suffix.lower() == ".wav":
            try:
                shutil.copy2(src, out_path)
                return out_path
            except Exception:
                # If copy fails for any reason, fall back to ffmpeg.
                pass

        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            raise RuntimeError(
                "ffmpeg was not found in PATH. Install ffmpeg (or add it to PATH) "
                "to generate preview_mix.wav from non-WAV audio."
            )

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(src),
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(out_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.is_file():
            stderr = (proc.stderr or "").strip()
            raise RuntimeError(
                f"ffmpeg conversion failed (code {proc.returncode}).\n{stderr}"
            )

        return out_path

    def get_cover_path(self) -> Optional[Path]:
        """Return absolute path to cover image, if any."""
        if not self.cover_file:
            return None
        return self.folder / self.cover_file

    def set_lyrics_text(self, text: str) -> None:
        """Store lyrics text in project metadata."""
        self.lyrics_text = text
        self.save()


# ----------------------------------------------------------------------
# Project CRUD helpers
# ----------------------------------------------------------------------

def list_projects() -> List[Project]:
    """
    List all projects under PROJECTS_ROOT.

    Sorted by "most recently updated" first (newest -> oldest). This matches
    typical user expectations in the UI: the project you worked on last should
    appear at the top.

    We primarily sort on ``updated_at`` and fall back to ``created_at`` when
    needed (older metadata or corrupted timestamps).
    """
    _init_storage()
    projects: List[Project] = []

    for sub in PROJECTS_ROOT.iterdir():
        if not sub.is_dir():
            continue
        meta_path = sub / "project.json"
        if not meta_path.is_file():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            project = Project(**data)
            projects.append(project)
        except Exception:
            # Corrupt / incompatible project: just skip
            continue

    def sort_key(p: Project) -> datetime:
        # Prefer updated_at; fall back to created_at; and finally to datetime.min.
        for field_name in ("updated_at", "created_at"):
            try:
                value = getattr(p, field_name, "") or ""
                if value:
                    return datetime.fromisoformat(value)
            except Exception:
                pass
        return datetime.min

    # Newest first
    projects.sort(key=sort_key, reverse=True)
    return projects


def load_project(folder_name: str) -> Project:
    """
    Load a project from a folder name.

    Mainly useful for tooling; the UI uses list_projects().
    """
    _init_storage()
    meta_path = PROJECTS_ROOT / folder_name / "project.json"
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Project(**data)


def create_project(name: str) -> Project:
    """
    Create a new project.

    - Internal 'id' is a UUID
    - 'project_number' is monotonically increasing (1, 2, 3, …)
    - Folder on disk is 'Project #n (Name)'
    """
    _init_storage()
    now = datetime.utcnow().isoformat()

    meta = _load_global_meta()
    next_number = int(meta.get("next_project_number", 1))
    project_number = next_number
    meta["next_project_number"] = project_number + 1
    _save_global_meta(meta)

    folder_name = _build_folder_name(project_number, name)

    project = Project(
        id=uuid.uuid4().hex,
        name=name,
        created_at=now,
        updated_at=now,
        project_number=project_number,
        folder_name=folder_name,
    )
    project.save()
    return project


def delete_project(project: Project) -> None:
    """Delete the project folder and all its contents."""
    if project.folder.is_dir():
        shutil.rmtree(project.folder, ignore_errors=True)
