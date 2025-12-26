from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa  # type: ignore[import]
except Exception:  # pragma: no cover - librosa is optional at runtime
    librosa = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt, QUrl, QTimer, QSettings
from PyQt6.QtGui import QColor, QPixmap, QFontDatabase
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QFormLayout,
    QCheckBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
    QColorDialog, 
    QSizePolicy,
    QToolButton,
    QFileDialog, 
)

from .project_manager import Project
from .visualization_api import PluginParameter
from .lyrics_visualization_api import LyricsFrameContext, BaseLyricsVisualization
from .lyrics_visualizations_manager import LyricsVisualizationsManager, LyricsVisualizationInfo
from .lyrics_text_style import text_style_parameters

@dataclass
class AudioEnvelope:
    """Precomputed RMS envelope used for live previews."""
    path: Path
    rms: np.ndarray
    duration: float
    fps: int


class LyricsVisualizationsTab(QWidget):
    """
    Tab for text-based (lyrics) visualizations.

    The tab lets the user:
      * select a lyrics visualization plugin,
      * configure its parameters through auto-generated widgets,
      * choose the audio source that drives amplitude (main mix or stems),
      * preview the result in sync with the global QMediaPlayer.
    """

    PREVIEW_FPS = 25  # used when computing RMS envelopes

    def __init__(self, player: QMediaPlayer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.player = player

        self._project: Optional[Project] = None
        self._phrases: List[Dict[str, Any]] = []
        self._words: List[Dict[str, Any]] = []

        # Cached cover pixmap for background modes that use the project cover.
        # It is (re)loaded whenever set_project(...) is called.
        self._cover_pixmap: Optional[QPixmap] = None

        # Plugin discovery / instantiation
        self._manager = LyricsVisualizationsManager()
        self._current_plugin_id: Optional[str] = None
        self._current_plugin: Optional[BaseLyricsVisualization] = None

        # Audio routing / analysis
        self._audio_source_id: str = "main"
        self._audio_envelope: Optional[AudioEnvelope] = None
        self._audio_cache: Dict[Tuple[Path, int], AudioEnvelope] = {}

        # Per-plugin config cache (kept in sync with Project.*_by_plugin fields)
        self._plugin_params_by_id: Dict[str, Dict[str, Any]] = {}
        self._plugin_routing_by_id: Dict[str, Dict[str, str]] = {}

        # ------------------------------------------------------------------
        # Global text-style settings (shared across plugins & projects)
        # ------------------------------------------------------------------
        self._settings = QSettings("Olaf", "OlafApp")

        # All parameter names that belong to the shared text-style block.
        # Start from the core text-style parameters (font, outline, shadow,
        # background box) defined in lyrics_text_style.text_style_parameters.
        self._text_style_param_names = set(text_style_parameters().keys())

        # Also treat background-related parameters from the lyrics plugins
        # as part of the shared "Global text and background style" block so that:
        #   - they appear in the same group in the UI,
        #   - they are persisted in QSettings and reused across plugins.
        self._text_style_param_names.update(
            {
                "background_mode",           # gradient / solid / cover
                "background_color",          # solid background color
                "background_gradient_top",   # top color of vertical gradient
                "background_gradient_bottom" # bottom color of vertical gradient
            }
        )


        # Whether the text-style block is collapsed (stored in QSettings)
        collapsed = self._settings.value("lyrics/text_style_collapsed", False, type=bool)
        self._text_style_collapsed = bool(collapsed)

        # UI-related attributes (assigned in _build_ui)
        self.project_label: QLabel
        self.plugin_combo: QComboBox
        self.btn_rescan_plugins: QPushButton
        self.plugin_description_label: QLabel
        self.parameters_group: QGroupBox
        self.text_style_toggle: QToolButton
        self.text_style_panel: QWidget
        self.text_style_form_layout: QFormLayout
        self.plugin_params_group: QGroupBox
        # plugin-specific parameters live here:
        self.parameter_form_layout: QFormLayout
        self.parameter_widgets: Dict[str, QWidget] = {}

        self.audio_source_combo: QComboBox
        self.btn_play_preview: QPushButton
        self.status_label: QLabel

        self.preview_container: QWidget
        self.preview_layout: QVBoxLayout

        self._build_ui()


    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        """
        Called by MainWindow when the selected project changes.
        """
        self._project = project
        self._phrases = []
        self._words = []
        self._audio_envelope = None

        # Load the project's cover pixmap (used by plugins when background_mode=='cover')
        self._cover_pixmap = None
        if project is not None:
            cover_rel = getattr(project, "cover_file", None)
            if cover_rel:
                cover_path = project.folder / cover_rel
                if cover_path.is_file():
                    pm = QPixmap(str(cover_path))
                    if not pm.isNull():
                        self._cover_pixmap = pm

        # Reset per-plugin caches whenever we switch project
        self._plugin_params_by_id = {}
        self._plugin_routing_by_id = {}

        if project is None:
            self.project_label.setText("Current project: (none)")
            self._clear_plugin_instance()
            self._clear_parameters_form()
            self._rebuild_audio_sources()
            return

        self.project_label.setText(f"Current project: {project.name}")

        # Load alignment (phrases + words) from disk, if available.
        self._load_alignment_from_disk(project)

        # Rebuild audio sources combo (main + stems)
        self._rebuild_audio_sources()

        # Refresh plugins list
        self._manager.refresh()
        self._rebuild_plugin_combo()

        # ---- Load project-level lyrics visual config ------------------
        pid = getattr(project, "lyrics_visual_plugin_id", None)
        params_global = getattr(project, "lyrics_visual_parameters", {}) or {}
        routing_global = getattr(project, "lyrics_visual_routing", {}) or {}

        # Per-plugin maps from the project (if present)
        params_by_plugin = getattr(
            project, "lyrics_visual_parameters_by_plugin", {}
        ) or {}
        routing_by_plugin = getattr(
            project, "lyrics_visual_routing_by_plugin", {}
        ) or {}

        # Normalize to dict[str, dict]
        self._plugin_params_by_id = {
            str(k): dict(v or {})
            for k, v in params_by_plugin.items()
            if isinstance(v, dict)
        }
        self._plugin_routing_by_id = {
            str(k): dict(v or {})
            for k, v in routing_by_plugin.items()
            if isinstance(v, dict)
        }

        # Backward compatibility: if we only have the old global fields,
        # seed the per-plugin dict for the stored plugin id.
        if pid and params_global and pid not in self._plugin_params_by_id:
            self._plugin_params_by_id[pid] = dict(params_global)
        if pid and routing_global and pid not in self._plugin_routing_by_id:
            self._plugin_routing_by_id[pid] = dict(routing_global)

        # Ensure the combo shows a valid plugin id
        if pid and self._manager.get_info(pid) is not None:
            self._select_plugin_in_combo(pid)
        elif self.plugin_combo.count() > 0:
            # Fallback: first plugin in the list, if any.
            self.plugin_combo.setCurrentIndex(0)
            pid = self.plugin_combo.currentData()
        else:
            pid = None

        self._current_plugin_id = pid

        # Determine initial config for the selected plugin
        initial_params: Dict[str, Any] = {}
        if pid and pid in self._plugin_params_by_id:
            initial_params = dict(self._plugin_params_by_id[pid])
        elif pid and pid == getattr(project, "lyrics_visual_plugin_id", None):
            # Backward-compatible: use old flat parameters if plugin matches
            initial_params = dict(params_global)

        # Instantiate plugin if we have a valid id
        # Fill missing shared style keys from QSettings (defaults) + migrate legacy values
        self._apply_global_text_style_defaults(initial_params)
        self._create_plugin_instance(pid, initial_config=initial_params)

        # Restore audio routing, preferring per-plugin routing if available
        audio_source_id = "main"
        if pid and pid in self._plugin_routing_by_id:
            audio_source_id = self._plugin_routing_by_id[pid].get("audio_source", "main")
        elif routing_global:
            audio_source_id = routing_global.get("audio_source", "main")

        self._audio_source_id = audio_source_id
        self._apply_audio_source_to_combo(audio_source_id)
        self._update_audio_envelope()

        # Build parameters UI for the active plugin
        info = self._manager.get_info(pid) if pid else None

        # Update plugin description in the header box when arriving on the tab
        if hasattr(self, "plugin_description_label"):
            if info is not None:
                self.plugin_description_label.setText(info.description or "")
            else:
                self.plugin_description_label.setText("")

        self._rebuild_parameter_controls(info)


    def on_position_changed(self, position_ms: int) -> None:
        """
        Called by MainWindow whenever the shared QMediaPlayer position changes.
        This drives the live lyrics preview.
        """
        if self._project is None:
            return
        if self._current_plugin is None:
            return
        if not self._phrases:
            # Without alignment data, there is nothing meaningful to display.
            return

        t = max(0.0, position_ms / 1000.0)

        amp = 0.0
        env = self._audio_envelope
        if env is not None and env.rms.size > 0:
            idx = int(t * env.fps)
            if idx < 0:
                idx = 0
            if idx >= env.rms.size:
                idx = env.rms.size - 1
            amp = float(env.rms[idx])

        ctx = self._build_context_for_time(t, amp)
        self._apply_runtime_font_scaling()
        self._current_plugin.update_frame(ctx)

    def on_duration_changed(self, duration_ms: int) -> None:
        """
        For now, we do not need to react to duration changes, but the method
        is provided for symmetry with other tabs and future extensions.
        """
        _ = duration_ms  # unused

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)

        # Top: current project (hidden, project is shown in global bar)
        self.project_label = QLabel("", self)
        self.project_label.setVisible(False)
        main_layout.addWidget(self.project_label)

        # Central area: preview on the left, parameters on the right
        center_layout = QHBoxLayout()
        main_layout.addLayout(center_layout, stretch=1)

        # --------------------------------------------------------------
        # LEFT: preview + overlay controls (plugin + audio routing)
        # --------------------------------------------------------------
        left_container = QWidget(self)
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        center_layout.addWidget(left_container, stretch=2)

        # Preview container: plugin widget is inserted here
        self.preview_container = QWidget(left_container)
        self.preview_layout = QVBoxLayout(self.preview_container)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.setSpacing(0)
        left_layout.addWidget(self.preview_container, stretch=1)

        # Overlay panel: sits on top of the preview, top-left corner
        # and contains both "Lyrics style plugin" and "Audio routing".
        self._overlay_panel = QWidget(self.preview_container)
        self._overlay_panel.setObjectName("LyricsOverlayPanel")
        self._overlay_panel.setStyleSheet(
            """
            #LyricsOverlayPanel {
                background-color: rgba(0, 0, 0, 170);
                border-radius: 8px;
            }
            """
        )
        # Horizontal layout so both groups sit side by side
        overlay_layout = QHBoxLayout(self._overlay_panel)
        overlay_layout.setContentsMargins(8, 8, 8, 8)
        overlay_layout.setSpacing(8)

        # --- Group: plugin selection + description --------------------
        plugin_group = QGroupBox("Lyrics style plugin", self._overlay_panel)
        plugin_layout = QVBoxLayout(plugin_group)
        plugin_layout.setContentsMargins(4, 4, 4, 4)
        plugin_layout.setSpacing(4)

        # First row: combo + rescan button
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        self.plugin_combo = QComboBox(plugin_group)
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_combo_changed)
        row_layout.addWidget(self.plugin_combo, stretch=1)

        self.btn_rescan_plugins = QPushButton("Rescan plugins", plugin_group)
        self.btn_rescan_plugins.clicked.connect(self._on_rescan_plugins_clicked)
        row_layout.addWidget(self.btn_rescan_plugins)

        plugin_layout.addLayout(row_layout)

        # Second row: plugin description
        self.plugin_description_label = QLabel("", plugin_group)
        self.plugin_description_label.setWordWrap(True)
        self.plugin_description_label.setObjectName("lyrics_plugin_description")
        self.plugin_description_label.setStyleSheet(
            "color: #bbbbbb; font-size: 10pt;"
        )
        plugin_layout.addWidget(self.plugin_description_label)

        plugin_group.setLayout(plugin_layout)

        # Keep for geometry adjustments and force horizontal expansion
        self._plugin_group = plugin_group
        plugin_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )


        # --- Group: audio routing -------------------------------------
        routing_group = QGroupBox("Audio routing", self._overlay_panel)
        routing_layout = QFormLayout(routing_group)

        self.audio_source_combo = QComboBox(routing_group)
        self.audio_source_combo.currentIndexChanged.connect(
            self._on_audio_source_changed
        )
        routing_layout.addRow("Audio source:", self.audio_source_combo)

        self.btn_play_preview = QPushButton("Play preview", routing_group)
        self.btn_play_preview.clicked.connect(self._on_play_preview_clicked)
        routing_layout.addRow(self.btn_play_preview)

        routing_group.setLayout(routing_layout)
        
        # Keep for geometry adjustments and force horizontal expansion
        self._routing_group = routing_group
        routing_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )


        # Put both groups side by side and let them share the available width
        overlay_layout.addWidget(plugin_group, 1)
        overlay_layout.addWidget(routing_group, 1)


        # --------------------------------------------------------------
        # RIGHT: plugin parameters + status (plus de groupbox globale)
        # --------------------------------------------------------------
        right_container = QWidget(self)
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        center_layout.addWidget(right_container, stretch=1)

        # Parameters group (contains a collapsible global text-style section
        # and a plugin-specific section).
        self.parameters_group = QGroupBox("Plugin parameters", right_container)
        params_layout = QVBoxLayout(self.parameters_group)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(6)

        # --- Collapsible "Global text style" section + presets ---------
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        self.text_style_toggle = QToolButton(self.parameters_group)
        self.text_style_toggle.setText("Global text and background style")

        self.text_style_toggle.setCheckable(True)
        # Expanded when not collapsed
        self.text_style_toggle.setChecked(not self._text_style_collapsed)
        self.text_style_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.text_style_toggle.setArrowType(
            Qt.ArrowType.DownArrow
            if not self._text_style_collapsed
            else Qt.ArrowType.RightArrow
        )
        self.text_style_toggle.clicked.connect(self._on_text_style_toggle_clicked)
        header_row.addWidget(self.text_style_toggle)

        header_row.addStretch(1)

        # Preset buttons: Save / Load / Reset
        self.btn_text_style_save = QToolButton(self.parameters_group)
        self.btn_text_style_save.setText("Save…")
        self.btn_text_style_save.setToolTip("Save current lyrics text-style as a preset")
        self.btn_text_style_save.clicked.connect(self._on_save_text_style_preset_clicked)
        header_row.addWidget(self.btn_text_style_save)

        self.btn_text_style_load = QToolButton(self.parameters_group)
        self.btn_text_style_load.setText("Load…")
        self.btn_text_style_load.setToolTip("Load a lyrics text-style preset from disk")
        self.btn_text_style_load.clicked.connect(self._on_load_text_style_preset_clicked)
        header_row.addWidget(self.btn_text_style_load)

        self.btn_text_style_reset = QToolButton(self.parameters_group)
        self.btn_text_style_reset.setText("Reset")
        self.btn_text_style_reset.setToolTip("Reset lyrics text-style to defaults")
        self.btn_text_style_reset.clicked.connect(self._on_reset_text_style_defaults_clicked)
        header_row.addWidget(self.btn_text_style_reset)

        params_layout.addLayout(header_row)

        self.text_style_panel = QWidget(self.parameters_group)
        self.text_style_form_layout = QFormLayout(self.text_style_panel)
        self.text_style_form_layout.setContentsMargins(8, 0, 8, 0)
        self.text_style_form_layout.setSpacing(4)
        params_layout.addWidget(self.text_style_panel)

        # --- Plugin-specific parameters --------------------------------
        self.plugin_params_group = QGroupBox("Effect parameters", self.parameters_group)
        self.parameter_form_layout = QFormLayout(self.plugin_params_group)
        self.plugin_params_group.setLayout(self.parameter_form_layout)
        params_layout.addWidget(self.plugin_params_group)

        right_layout.addWidget(self.parameters_group, stretch=1)

        # Initial visibility of the global text-style block
        self._update_text_style_panel_visibility()


        # Status label
        self.status_label = QLabel("", right_container)
        right_layout.addWidget(self.status_label)

        right_layout.addStretch(1)
        self.setLayout(main_layout)

    # ------------------------------------------------------------------
    # Alignment loading and context building
    # ------------------------------------------------------------------
    def _update_overlay_geometry(self) -> None:
        """
        Position the overlay panel (plugin + routing) in the top-left
        corner of the preview container.

        On laisse QHBoxLayout gérer le partage 50/50 entre les deux
        QGroupBox via les facteurs de stretch (1, 1).
        """
        if not hasattr(self, "_overlay_panel"):
            return
        if self.preview_container is None:
            return

        rect = self.preview_container.rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        margin = 12

        # L'overlay prend quasiment toute la largeur du preview
        overlay_width = max(260, rect.width() - 2 * margin)
        overlay_height = self._overlay_panel.sizeHint().height()

        self._overlay_panel.setGeometry(
            margin,
            margin,
            overlay_width,
            overlay_height,
        )

    def resizeEvent(self, event) -> None:
        """
        Ensure the overlay (lyrics plugin + audio routing) stays anchored
        in the top-left corner of the preview when the tab is resized.
        """
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def showEvent(self, event) -> None:
        """
        When the tab is first shown, the layout has its final size.
        We schedule an overlay geometry update on the next event loop
        iteration to avoid using the tiny initial rect().
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._update_overlay_geometry)

    def _load_alignment_from_disk(self, project: Project) -> None:
        """
        Load existing alignment (phrases + words) from project/vocal_align/*.json
        if available, so timings survive across sessions.
        """
        align_dir = project.folder / "vocal_align"
        phrases_path = align_dir / "phrases.json"
        words_path = align_dir / "words.json"

        self._phrases = []
        self._words = []

        if not phrases_path.is_file():
            # No previous alignment saved for this project
            self.status_label.setText("No alignment found (phrases.json missing).")
            return

        try:
            phrases_data = json.loads(phrases_path.read_text(encoding="utf-8"))
            if isinstance(phrases_data, list):
                self._phrases = phrases_data
        except Exception as e:
            self._phrases = []
            self.status_label.setText(f"Could not read phrases.json: {e}")
            return

        if words_path.is_file():
            try:
                words_data = json.loads(words_path.read_text(encoding="utf-8"))
                if isinstance(words_data, list):
                    self._words = words_data
            except Exception as e:
                self._words = []
                self.status_label.setText(f"Could not read words.json: {e}")
        else:
            self._words = []

        # On success, we now stay silent to save UI space.
        # Errors and "no alignment" cases are still reported via status_label.


    def _build_context_for_time(self, t: float, amp: float) -> LyricsFrameContext:
        """
        Map a playback time (seconds) to the current phrase + word using the
        alignment data loaded from disk (phrases.json + words.json).

        Key requirements:
          - repeated words on a line must highlight the correct occurrence ("the ... the")
          - edits may reorder the internal list; do NOT rely on raw list order
          - some words may have no word_index (e.g. inserted manually): fallback to time order
        """
        phrase_index: Optional[int] = None
        phrase_obj: Optional[Dict[str, Any]] = None
        phrase_start = 0.0
        phrase_end = 0.0

        # 1) Find active phrase
        for idx, phrase in enumerate(self._phrases or []):
            try:
                start = float(phrase.get("start", 0.0))
                end = float(phrase.get("end", start))
            except Exception:
                continue

            if start <= t <= end:
                phrase_index = idx
                phrase_obj = phrase
                phrase_start = start
                phrase_end = end
                break

        if phrase_obj is None:
            return LyricsFrameContext(
                t=t,
                amp=amp,
                phrase_index=None,
                local_phrase_time=0.0,
                phrase_duration=0.0,
                text_full_line="",
                word_index=None,
                text_active_word=None,
                word_char_start=None,
                word_char_end=None,
            )

        text_full_line = str(phrase_obj.get("text", ""))
        local_phrase_time = max(0.0, t - phrase_start)
        phrase_duration = max(0.0, phrase_end - phrase_start)

        # 2) Collect words for this line
        line_index = phrase_obj.get("line_index", phrase_index)

        raw_line_words: list[tuple[int, Dict[str, Any]]] = []
        for gidx, w in enumerate(self._words or []):
            if w.get("line_index", line_index) != line_index:
                continue
            raw_line_words.append((gidx, w))

        def _safe_f(x: Any, default: float = 0.0) -> float:
            try:
                return float(x)
            except Exception:
                return float(default)

        def _safe_i(x: Any, default: int = -1) -> int:
            try:
                return int(x)
            except Exception:
                return int(default)

        # Build a stable time order for this line (also used as fallback "text order").
        by_time = sorted(
            raw_line_words,
            key=lambda it: (
                _safe_f(it[1].get("start", 0.0), 0.0),
                _safe_f(it[1].get("end", it[1].get("start", 0.0)), 0.0),
                it[0],
            ),
        )
        time_rank: dict[int, int] = {gidx: i for i, (gidx, _) in enumerate(by_time)}

        # Text order: prefer word_index when present; fallback to time order if missing.
        by_text = sorted(
            raw_line_words,
            key=lambda it: (
                (_safe_i(it[1].get("word_index", -1), -1) if _safe_i(it[1].get("word_index", -1), -1) >= 0 else time_rank.get(it[0], 10**9)),
                time_rank.get(it[0], 10**9),
                it[0],
            ),
        )

        # 3) Find active word by timing (robust to overlaps)
        current_word_text: Optional[str] = None
        current_word_global_idx: Optional[int] = None

        candidates: list[tuple[float, float, int, Dict[str, Any]]] = []
        for gidx, w in by_time:
            ws = _safe_f(w.get("start", 0.0), 0.0)
            we = _safe_f(w.get("end", ws), ws)
            if ws <= t <= we:
                candidates.append((ws, we, gidx, w))

        if candidates:
            # Prefer the word that started most recently; then shortest end.
            ws, we, gidx, w = sorted(candidates, key=lambda it: (-it[0], it[1], it[2]))[0]
            _ = (ws, we)
            current_word_text = str(w.get("text", ""))
            current_word_global_idx = int(gidx)

        # 4) Map active word -> character offsets in the phrase (token-aware + repeated words)
        word_char_start: Optional[int] = None
        word_char_end: Optional[int] = None

        def _is_word_char(ch: str) -> bool:
            return ch.isalnum() or ch in ("_", "-", "'", "’")

        def _find_nth_exact_token(haystack: str, needle: str, n: int) -> int:
            """Return the start index of the n-th exact token occurrence, or -1."""
            if not needle:
                return -1

            count = 0
            i = 0
            L = len(needle)

            while True:
                j = haystack.find(needle, i)
                if j < 0:
                    return -1

                left_ok = (j == 0) or (not _is_word_char(haystack[j - 1]))
                right_i = j + L
                right_ok = (right_i >= len(haystack)) or (not _is_word_char(haystack[right_i]))

                if left_ok and right_ok:
                    if count == n:
                        return j
                    count += 1

                i = j + L

        if current_word_text and current_word_global_idx is not None:
            lower_line = text_full_line.lower()
            lower_word = current_word_text.lower()

            # Count how many identical tokens appear before this one, in TEXT order.
            occurrence_index = 0
            for gidx, w in by_text:
                if gidx == current_word_global_idx:
                    break
                if str(w.get("text", "")).lower() == lower_word:
                    occurrence_index += 1

            pos = _find_nth_exact_token(lower_line, lower_word, occurrence_index)
            if pos >= 0:
                word_char_start = pos
                word_char_end = pos + len(lower_word)

        return LyricsFrameContext(
            t=t,
            amp=amp,
            phrase_index=phrase_index,
            local_phrase_time=local_phrase_time,
            phrase_duration=phrase_duration,
            text_full_line=text_full_line,
            word_index=current_word_global_idx,
            text_active_word=current_word_text,
            word_char_start=word_char_start,
            word_char_end=word_char_end,
        )

    # ------------------------------------------------------------------
    # Plugin combo / instance management
    # ------------------------------------------------------------------
    def _rebuild_plugin_combo(self) -> None:
        """Populate the plugin combo box from the manager registry."""
        self.plugin_combo.blockSignals(True)
        self.plugin_combo.clear()

        visuals = self._manager.available_visuals()
        for info in visuals:
            self.plugin_combo.addItem(info.name, info.plugin_id)

        self.plugin_combo.blockSignals(False)

        # Clear description until a plugin is effectively selected
        if hasattr(self, "plugin_description_label"):
            self.plugin_description_label.setText("")

    def _select_plugin_in_combo(self, plugin_id: str) -> None:
        """
        Select a plugin in the combo box by its id, if present.
        """
        self.plugin_combo.blockSignals(True)
        index_to_select = -1
        for i in range(self.plugin_combo.count()):
            if self.plugin_combo.itemData(i) == plugin_id:
                index_to_select = i
                break
        if index_to_select >= 0:
            self.plugin_combo.setCurrentIndex(index_to_select)
        self.plugin_combo.blockSignals(False)

    def _create_plugin_instance(
        self,
        plugin_id: Optional[str],
        initial_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiate the selected plugin and place its widget in the preview area."""
        self._clear_plugin_instance()

        if not plugin_id:
            self._current_plugin = None
            self._current_plugin_id = None
            return

        info = self._manager.get_info(plugin_id)
        if info is None:
            self._current_plugin = None
            self._current_plugin_id = None
            return

        # Create the plugin widget
        instance = self._manager.create_instance(
            plugin_id, config=initial_config or {}, parent=self.preview_container
        )
        if instance is None:
            QMessageBox.critical(
                self,
                "Plugin error",
                f"Failed to instantiate lyrics plugin '{plugin_id}'.",
            )
            self._current_plugin = None
            self._current_plugin_id = None
            return

        # Apply global style defaults (QSettings) and migrate legacy font sizing
        cfg0 = getattr(instance, "config", None)
        if isinstance(cfg0, dict):
            self._apply_global_text_style_defaults(cfg0)

        # Simplified background behaviour: always use the project cover
        # (if the plugin exposes a "background_mode" parameter, it is
        # forced to "cover" so that gradient / solid modes are disabled).
        cfg = getattr(instance, "config", None)
        if isinstance(cfg, dict) and "background_mode" in cfg:
            cfg["background_mode"] = "cover"


        # --------------------------------------------------------------
        # IMPORTANT: Inject project cover pixmap into plugin instance.
        # Plugins check this attribute when background_mode == 'cover'.
        # --------------------------------------------------------------
        if self._cover_pixmap is not None:
            instance.cover_pixmap = self._cover_pixmap

        # Add plugin widget into preview container
        self.preview_layout.addWidget(instance)
        instance.show()

        self._current_plugin = instance
        self._current_plugin_id = plugin_id

        # Ensure the overlay (plugin selection + audio routing) stays visible
        # on top of the preview widget.
        if hasattr(self, "_overlay_panel"):
            try:
                self._overlay_panel.raise_()
                self._update_overlay_geometry()
            except Exception:
                # Do not let UI crash if overlay geometry fails for any reason.
                pass


    def _clear_plugin_instance(self) -> None:
        """Remove any existing plugin widget from the preview area."""
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._current_plugin = None
        self._current_plugin_id = None

    # ------------------------------------------------------------------
    # Plugin parameters widgets
    # ------------------------------------------------------------------
    def _clear_parameters_form(self) -> None:
        """
        Remove all parameter rows from both the global text-style section
        and the plugin-specific section.
        """
        for layout in (self.text_style_form_layout, self.parameter_form_layout):
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()
        self.parameter_widgets.clear()

    def _rebuild_parameter_controls(self, info: Optional[LyricsVisualizationInfo]) -> None:
        """
        Rebuild the parameter widgets list according to the plugin's
        declared parameters.

        Parameters whose names are listed in self._text_style_param_names
        go into the collapsible "Global text and background style" section.

        All other parameters go into the "Effect parameters" group.

        Background parameters controlling gradient / solid / cover
        behaviour are intentionally hidden from the UI: lyrics visuals
        now always use the project cover as background.
        """
        # Clear both global text-style and plugin-specific sections
        self._clear_parameters_form()

        if info is None or not info.parameters:
            placeholder = QLabel(
                "This plugin does not declare any configurable parameters.",
                self.parameters_group,
            )
            placeholder.setWordWrap(True)
            self.parameter_form_layout.addRow(placeholder)
            return

        # Use the plugin instance config as the current value source
        plugin_config: Dict[str, Any] = {}
        if (
            self._current_plugin is not None
            and info.plugin_id == self._current_plugin_id
        ):
            plugin_config = getattr(self._current_plugin, "config", {}) or {}

        # Parameters that are no longer exposed in the UI because the
        # background is always driven by the project cover.
        hidden_background_params = {
            "background_mode",
            "background_color",
            "background_gradient_top",
            "background_gradient_bottom",
        }

        for name, param in info.parameters.items():
            # Skip legacy background controls (we always use cover)
            if name in hidden_background_params:
                continue

            current_value = plugin_config.get(name, param.default)
            widget = self._create_widget_for_parameter(param, current_value)
            self.parameter_widgets[name] = widget
            label = QLabel(param.label or name, self.parameters_group)

            # Route to the correct section:
            # - global text/background style -> collapsable panel
            # - everything else -> Effect parameters
            if name in self._text_style_param_names:
                self.text_style_form_layout.addRow(label, widget)
            else:
                self.parameter_form_layout.addRow(label, widget)

            self._connect_parameter_widget(name, param, widget)

        # Ensure the collapsable section respects the stored expanded/collapsed state
        self._update_text_style_panel_visibility()

    def _on_text_style_toggle_clicked(self) -> None:
        """
        Handle user toggling the 'Global text style' header.

        We store the collapsed/expanded state in QSettings so that the
        UI behaves consistently across sessions.
        """
        expanded = self.text_style_toggle.isChecked()
        self._settings.setValue("lyrics/text_style_collapsed", not expanded)
        self._update_text_style_panel_visibility()

    def _update_text_style_panel_visibility(self) -> None:
        """
        Show/hide the global text-style parameter panel and update the arrow.
        """
        expanded = self.text_style_toggle.isChecked()
        self.text_style_panel.setVisible(expanded)
        self.text_style_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _on_save_text_style_preset_clicked(self) -> None:
        """
        Save the current global lyrics text-style parameters (including
        background-related ones) to a JSON file chosen by the user.
        """
        if self._current_plugin is None:
            QMessageBox.information(
                self,
                "No plugin",
                "No lyrics plugin is active; nothing to save.",
            )
            return

        cfg = dict(getattr(self._current_plugin, "config", {}) or {})
        data: Dict[str, Any] = {}
        for name in self._text_style_param_names:
            if name in cfg:
                data[name] = cfg[name]

        if not data:
            QMessageBox.warning(
                self,
                "No data",
                "There are no text-style parameters to save.",
            )
            return

        start_dir = Path.home()
        if self._project is not None:
            start_dir = self._project.folder

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save lyrics text-style preset",
            str(start_dir / "lyrics_text_style.json"),
            "JSON files (*.json)",
        )
        if not file_path:
            return

        try:
            Path(file_path).write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Save error",
                f"Could not save preset:\n{exc}",
            )

    def _on_load_text_style_preset_clicked(self) -> None:
        """
        Load a lyrics text-style preset from a JSON file and apply it to
        the current plugin + shared settings.
        """
        start_dir = Path.home()
        if self._project is not None:
            start_dir = self._project.folder

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load lyrics text-style preset",
            str(start_dir),
            "JSON files (*.json)",
        )
        if not file_path:
            return

        try:
            raw = Path(file_path).read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load error",
                f"Could not read preset:\n{exc}",
            )
            return

        if not isinstance(data, dict):
            QMessageBox.warning(
                self,
                "Invalid preset",
                "The selected file does not contain a JSON object.",
            )
            return

        self._apply_text_style_from_dict(data)

    def _on_reset_text_style_defaults_clicked(self) -> None:
        """
        Reset all shared text-style parameters (font, outline, shadow,
        background box and background mode/colors) to their default values.
        """
        # Core defaults from lyrics_text_style
        defaults: Dict[str, Any] = {}
        base_defs = text_style_parameters()
        for name, param in base_defs.items():
            defaults[name] = param.default

        # Plugin-level defaults for additional background parameters
        info = None
        if self._current_plugin_id is not None:
            info = self._manager.get_info(self._current_plugin_id)

        if info is not None:
            for name in self._text_style_param_names:
                if name in defaults:
                    continue
                plugin_param = info.parameters.get(name)
                if plugin_param is not None:
                    defaults[name] = plugin_param.default

        # Clear QSettings overrides for these parameters
        for name in self._text_style_param_names:
            try:
                self._settings.remove(f"lyrics/text_style/{name}")
            except Exception:
                pass

        # Apply defaults through the widgets (this will also update
        # plugin config + project JSON via the normal callbacks).
        self._apply_text_style_from_dict(defaults)

    def _apply_text_style_from_dict(self, values: Dict[str, Any]) -> None:
        """
        Apply a set of text-style values to the current UI controls and
        plugin configuration.

        This method drives the actual Qt widgets so that:
          - the UI is updated,
          - the usual callbacks (_on_parameter_changed, etc.) are triggered,
          - QSettings and project JSON stay in sync.
        """
        if not values:
            return

        for name, value in values.items():
            if name not in self._text_style_param_names:
                continue

            widget = self.parameter_widgets.get(name)
            if widget is None:
                continue

            # Color button (QPushButton with _color_name property)
            if isinstance(widget, QPushButton) and widget.property("_color_name") is not None:
                col_str = str(value)
                if not QColor.isValidColor(col_str):
                    continue
                widget.setProperty("_color_name", col_str)
                widget.setStyleSheet(f"background-color: {col_str}; border: 1px solid #555;")
                # Update plugin config + project
                self._on_parameter_changed(name, col_str)
                continue

            # Bool parameter
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
                continue

            # Enum parameter
            if isinstance(widget, QComboBox):
                target_index = 0
                for i in range(widget.count()):
                    if widget.itemData(i) == value or widget.itemText(i) == str(value):
                        target_index = i
                        break
                widget.setCurrentIndex(target_index)
                continue

            # Text parameter
            if isinstance(widget, QLineEdit):
                widget.setText("" if value is None else str(value))
                continue

            # Slider container (int / float)
            slider = getattr(widget, "_slider", None)
            if isinstance(slider, QSlider):
                min_v = getattr(widget, "_min", 0)
                step = getattr(widget, "_step", 1)

                # int slider
                if isinstance(min_v, int):
                    try:
                        v = int(value)
                    except Exception:
                        continue
                    v = max(min_v, min(v, getattr(widget, "_max", v)))
                    slider.setValue(v)
                else:
                    # float slider mapped to integer ticks
                    try:
                        v = float(value)
                    except Exception:
                        continue
                    min_f = float(min_v)
                    step_f = float(step)
                    max_f = float(getattr(widget, "_max", min_f))
                    if max_f < min_f:
                        max_f = min_f
                    if v < min_f:
                        v = min_f
                    if v > max_f:
                        v = max_f
                    index = int(round((v - min_f) / step_f))
                    slider.setValue(index)


    def _create_widget_for_parameter(
            self,
            param: PluginParameter,
            current_value: Any,
        ) -> QWidget:
        """
        Create a Qt widget for a given PluginParameter and initial value.

        This is the lyrics-specific variant of the 3D visualization UI builder.
        It supports:
          - bool   -> QCheckBox
          - enum   -> QComboBox
          - color  -> QPushButton opening a color dialog
          - str    -> QLineEdit
          - int    -> QSlider with integer mapping
          - float  -> QSlider with float mapping

        For the special "font_family" enum, the choice list is rebuilt from
        QFontDatabase and explicitly enriched with any fonts found in the
        project-level "fonts" directory (e.g. E:/Olaf/fonts), so that custom
        fonts are always visible in the combo box.
        """
        # Color parameter
        if param.type == "color":
            btn = QPushButton(self.parameters_group)
            btn.setText("")  # color-only button
            btn.setFixedWidth(60)

            # Current value or default
            col_str = str(
                current_value
                if current_value is not None
                else (param.default or "#ffffff")
            )
            if not QColor.isValidColor(col_str):
                col_str = "#ffffff"

            btn.setProperty("_color_name", col_str)
            btn.setStyleSheet(
                f"background-color: {col_str}; border: 1px solid #555;"
            )

            # Connect click to a color dialog handler
            btn.clicked.connect(
                lambda _checked=False, n=param.name, b=btn: self._on_color_button_clicked(n, b)
            )
            return btn

        # Bool parameter
        if param.type == "bool":
            w = QCheckBox(self.parameters_group)
            try:
                w.setChecked(bool(current_value))
            except Exception:
                w.setChecked(bool(param.default))
            return w

        # Enum parameter (choice list)
        if param.type == "enum":
            w = QComboBox(self.parameters_group)

            # Default: use choices declared by the plugin
            choices = list(param.choices or [])

            # Special case for font family: rebuild the list to include
            # *all* available families + fonts from /fonts at project root.
            if param.name == "font_family":
                family_set = set()

                # 1) Start from all families known by QFontDatabase
                try:
                    db = QFontDatabase()
                    for fam in db.families():
                        family_set.add(str(fam))
                except Exception:
                    # If this fails, we will at least keep plugin-provided choices
                    family_set.update(str(c) for c in choices)

                # 2) Explicitly scan project_root/fonts (NOT olaf_app/fonts)
                try:
                    here = Path(__file__).resolve()
                    project_root = here.parent.parent  # .../Olaf
                    fonts_dir = project_root / "fonts"

                    if fonts_dir.is_dir():
                        for pattern in ("*.ttf", "*.otf"):
                            for font_path in fonts_dir.glob(pattern):
                                try:
                                    font_id = QFontDatabase.addApplicationFont(
                                        str(font_path)
                                    )
                                    if font_id >= 0:
                                        for fam in QFontDatabase.applicationFontFamilies(font_id):
                                            family_set.add(str(fam))
                                except Exception:
                                    # Never let a broken font file or bad font
                                    # path crash the UI.
                                    pass
                except Exception:
                    # If path resolution fails for some reason, we still have
                    # whatever was gathered above.
                    pass

                if family_set:
                    choices = sorted(family_set)

            for choice in choices:
                w.addItem(str(choice), choice)

            index_to_select = 0
            for i in range(w.count()):
                if w.itemData(i) == current_value:
                    index_to_select = i
                    break
            w.setCurrentIndex(index_to_select)
            return w

        # Free text
        if param.type == "str":
            w = QLineEdit(self.parameters_group)
            if current_value is None:
                current_value = param.default
            if current_value is not None:
                w.setText(str(current_value))
            return w

        # Numeric parameters: int / float -> slider + value label
        container = QWidget(self.parameters_group)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal, container)
        slider.setMinimum(0)

        value_label = QLabel(container)
        value_label.setMinimumWidth(60)

        # Integer parameter
        if param.type == "int":
            min_v = int(param.minimum if param.minimum is not None else 0)
            max_v = int(param.maximum if param.maximum is not None else 100)
            if max_v < min_v:
                max_v = min_v

            step = int(param.step if param.step is not None else 1)
            if step <= 0:
                step = 1

            # Map [min_v, max_v] in steps of "step" to integer slider ticks
            num_steps = max(1, int(round((max_v - min_v) / step)))
            slider.setMaximum(num_steps)

            try:
                value = int(current_value)
            except Exception:
                value = int(param.default if param.default is not None else min_v)

            # Clamp and map to index
            if value < min_v:
                value = min_v
            if value > max_v:
                value = max_v

            index = int(round((value - min_v) / step))
            slider.setValue(index)
            value_label.setText(str(value))

            # Used by _on_slider_changed to distinguish int vs float
            setattr(container, "_min", int(min_v))
            setattr(container, "_max", int(max_v))
            setattr(container, "_step", int(step))

        else:
            # Float parameter
            min_v = float(param.minimum if param.minimum is not None else 0.0)
            max_v = float(param.maximum if param.maximum is not None else 1.0)
            if max_v < min_v:
                max_v = min_v

            step = float(param.step if param.step is not None else 0.01)
            if step <= 0.0:
                step = 0.01

            # We map [min_v, max_v] to discrete float steps
            num_steps = max(1, int(round((max_v - min_v) / step)))
            slider.setMaximum(num_steps)

            try:
                value = float(current_value)
            except Exception:
                value = float(param.default if param.default is not None else min_v)

            # Clamp to [min_v, max_v]
            if value < min_v:
                value = min_v
            if value > max_v:
                value = max_v

            # Map value back to slider index
            index = int(round((value - min_v) / step))
            slider.setValue(index)
            value_label.setText(f"{value:.3g}")

            # Store bounds for _on_slider_changed
            setattr(container, "_min", float(min_v))
            setattr(container, "_max", float(max_v))
            setattr(container, "_step", float(step))

        # Wire the slider to the generic handler
        slider.valueChanged.connect(
            lambda idx, c=container, n=param.name, lbl=value_label: self._on_slider_changed(
                c, idx, n, lbl
            )
        )

        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        return container

    def _connect_parameter_widget(
        self,
        name: str,
        param: PluginParameter,
        widget: QWidget,
    ) -> None:
        """
        Connect a parameter widget so that changes are propagated to the
        current plugin instance configuration.
        """
        # Bool parameter
        if isinstance(widget, QCheckBox):
            widget.toggled[bool].connect(
                lambda checked, n=name: self._on_parameter_changed(n, checked)
            )
            return

        # Enum parameter
        if isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(
                lambda _idx, w=widget, n=name: self._on_parameter_changed(
                    n, w.currentData()
                )
            )
            return

        # Text parameter
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(
                lambda text, n=name: self._on_parameter_changed(n, text)
            )
            return

        # Slider container for int/float
        slider = getattr(widget, "_slider", None)
        value_label = getattr(widget, "_value_label", None)
        if isinstance(slider, QSlider):
            slider.valueChanged.connect(
                lambda index, w=widget, n=name, label=value_label: self._on_slider_changed(
                    w, index, n, label
                )
            )
            
    def _on_color_button_clicked(self, name: str, button: QPushButton) -> None:
        """
        Open a QColorDialog to edit a color parameter and store the resulting
        hex string back into the plugin configuration.
        """
        current_name = button.property("_color_name")
        if not isinstance(current_name, str) or not QColor.isValidColor(current_name):
            current_name = "#ffffff"

        initial = QColor(current_name)
        color = QColorDialog.getColor(initial, self, "Choose color")
        if not color.isValid():
            return

        hex_name = color.name()  # "#rrggbb"
        button.setProperty("_color_name", hex_name)
        button.setStyleSheet(f"background-color: {hex_name}; border: 1px solid #555;")

        # Persist in plugin configuration and project JSON
        self._on_parameter_changed(name, hex_name)

    def _on_slider_changed(
        self,
        container: QWidget,
        index: int,
        name: str,
        value_label: Optional[QLabel],
    ) -> None:
        """Handle int/float slider movement.

        For integer parameters we map the discrete slider index back to the
        real value using the stored min / max / step. This keeps the actual
        parameter value consistent with the range declared in PluginParameter.
        """
        # Decide if int or float based on stored bounds
        min_v = getattr(container, "_min", 0)
        step = getattr(container, "_step", 1)

        if isinstance(min_v, int):
            # Integer parameter: map slider index -> value in [min_v, max_v]
            max_v = getattr(container, "_max", min_v)
            step_i = int(step) if step else 1
            if step_i <= 0:
                step_i = 1

            value = int(min_v + index * step_i)
            if value < min_v:
                value = min_v
            if value > int(max_v):
                value = int(max_v)

            if value_label is not None:
                value_label.setText(str(value))
        else:
            # Float parameter: map slider index -> value using min / step
            min_f = float(min_v)
            step_f = float(step) if step else 0.01
            if step_f <= 0.0:
                step_f = 0.01

            value = float(min_f + index * step_f)
            if value_label is not None:
                value_label.setText(f"{value:.3g}")

        self._on_parameter_changed(name, value)


    def _on_parameter_changed(self, name: str, value: Any) -> None:
        """
        Update the configuration dictionary of the active plugin instance
        when a parameter widget changes, and persist to the project.

        Shared text-style parameters are additionally stored in QSettings
        so they can be reused across lyrics plugins and projects.
        """
        if self._current_plugin is None:
            return

        try:
            self._current_plugin.config[name] = value
        except Exception:
            # Do not let plugins crash the host because of config handling.
            return

        # If this is one of the shared text-style parameters, store it globally.
        if name in getattr(self, "_text_style_param_names", set()):
            try:
                self._settings.setValue(f"lyrics/text_style/{name}", value)
            except Exception:
                pass

        self._save_to_project()


    # ------------------------------------------------------------------
    # Audio routing helpers
    # ------------------------------------------------------------------
    def _rebuild_audio_sources(self) -> None:
        """
        Populate the audio source combo with all available choices for the
        current project (main mix + stems).
        """
        self.audio_source_combo.blockSignals(True)
        self.audio_source_combo.clear()

        sources: List[tuple[str, str]] = [("Project main audio", "main")]

        if self._project is not None:
            stems_by_model = getattr(self._project, "stems_by_model", {}) or {}
            stem_names = set()
            for stems in stems_by_model.values():
                for stem_name in stems.keys():
                    stem_names.add(stem_name)

            for stem_name in sorted(stem_names):
                sources.append((f"{stem_name} stem", f"stem:{stem_name}"))

        for label, src_id in sources:
            self.audio_source_combo.addItem(label, src_id)

        self.audio_source_combo.blockSignals(False)

    def _apply_audio_source_to_combo(self, source_id: str) -> None:
        """Select the given source_id in the audio source combo, if possible."""
        self.audio_source_combo.blockSignals(True)
        index_to_select = 0
        for i in range(self.audio_source_combo.count()):
            if self.audio_source_combo.itemData(i) == source_id:
                index_to_select = i
                break
        self.audio_source_combo.setCurrentIndex(index_to_select)
        self.audio_source_combo.blockSignals(False)

    def _resolve_audio_path_for_source(self, source_id: str) -> Optional[Path]:
        """
        Resolve the audio file path for a given source_id:
          - 'main'      -> project's main audio file
          - 'stem:xxxx' -> search stem 'xxxx' in project.stems_by_model
        """
        if self._project is None:
            return None

        # Main audio file
        if source_id == "main":
            try:
                return self._project.get_audio_path()
            except Exception:
                return None

        # Stems
        if source_id.startswith("stem:"):
            stem_name = source_id.split(":", 1)[1]
            stems_by_model = getattr(self._project, "stems_by_model", {}) or {}
            for stems in stems_by_model.values():
                rel_path = stems.get(stem_name)
                if not rel_path:
                    continue
                candidate = self._project.folder / rel_path
                if candidate.is_file():
                    return candidate

        return None

    def _ensure_audio_envelope(self, audio_path: Path, fps: int) -> AudioEnvelope:
        """
        Compute (or reuse from cache) a normalized RMS envelope for a given
        audio file at a given frame rate (fps).
        """
        key = (audio_path.resolve(), fps)
        if key in self._audio_cache:
            return self._audio_cache[key]

        if librosa is None:
            raise RuntimeError("librosa is not available")

        # Load mono audio at a fixed sample rate (e.g. 22050 Hz).
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration = float(len(y) / sr)

        # Hop length ~ 1 frame per preview frame (fps).
        hop_length = max(1, int(sr / fps))

        # RMS envelope (librosa returns shape (1, n_frames)).
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        rms = rms.astype(np.float32)

        # Normalize to [0, 1].
        max_val = float(rms.max()) if rms.size > 0 else 0.0
        if max_val > 0.0:
            rms /= max_val
        else:
            rms[:] = 0.0

        env = AudioEnvelope(path=audio_path, rms=rms, duration=duration, fps=fps)
        self._audio_cache[key] = env
        return env

    def _update_audio_envelope(self) -> None:
        """Recompute or fetch the RMS envelope for the current audio source."""
        if self._project is None:
            self._audio_envelope = None
            return

        audio_path = self._resolve_audio_path_for_source(self._audio_source_id)
        if audio_path is None:
            self._audio_envelope = None
            self.status_label.setText("No audio file found for the selected source.")
            return

        try:
            self._audio_envelope = self._ensure_audio_envelope(
                audio_path, fps=self.PREVIEW_FPS
            )
            # Do not overwrite the status label here: the audio envelope
            # is prepared silently to avoid noisy messages in the UI.
            # (The analysis function itself is unchanged.)
            # self.status_label.setText(
            #     f"Audio envelope ready ({self._audio_envelope.fps} fps)."
            # )
        except Exception as exc:
            self._audio_envelope = None
            QMessageBox.critical(
                self,
                "Audio analysis error",
                f"Could not compute audio envelope:\n{exc}",
            )


    # ------------------------------------------------------------------
    # Persist configuration back into the Project dataclass
    # ------------------------------------------------------------------
    def _save_to_project(self) -> None:
        """
        Mirror current lyrics visualization config back into the Project
        dataclass and save it to disk.

        This updates both the legacy flat fields
            - lyrics_visual_plugin_id
            - lyrics_visual_parameters
            - lyrics_visual_routing
        and the new per-plugin dictionaries
            - lyrics_visual_parameters_by_plugin
            - lyrics_visual_routing_by_plugin
        so that each plugin keeps its own settings.
        """
        if self._project is None:
            return

        try:
            pid = self._current_plugin_id
            self._project.lyrics_visual_plugin_id = pid

            # Keep per-plugin parameters in memory
            if pid and self._current_plugin is not None:
                cfg = dict(getattr(self._current_plugin, "config", {}) or {})
                # Do not persist runtime-only `font_size` when using relative sizing.
                if "font_size_rel" in cfg:
                    cfg.pop("font_size", None)
                self._plugin_params_by_id[pid] = cfg

            # Keep per-plugin routing in memory
            if pid:
                self._plugin_routing_by_id.setdefault(pid, {})
                self._plugin_routing_by_id[pid]["audio_source"] = self._audio_source_id

            # Legacy flat fields: "current plugin" only (for backward compatibility)
            if pid and self._current_plugin is not None:
                self._project.lyrics_visual_parameters = dict(
                    getattr(self._current_plugin, "config", {}) or {}
                )
            else:
                self._project.lyrics_visual_parameters = {}

            self._project.lyrics_visual_routing = {
                "audio_source": self._audio_source_id
            }

            # New fields: full per-plugin maps
            self._project.lyrics_visual_parameters_by_plugin = dict(
                self._plugin_params_by_id
            )
            self._project.lyrics_visual_routing_by_plugin = dict(
                self._plugin_routing_by_id
            )

            self._project.save()
        except Exception:
            # Never crash UI because of save issues; silently ignore for now.
            pass

    
    # ------------------------------------------------------------------
    # Text-style helpers (Option A: relative font size)
    # ------------------------------------------------------------------
    def _apply_global_text_style_defaults(self, config: Dict[str, Any]) -> None:
        """
        Apply global text-style defaults stored in QSettings to *config*.

        QSettings values are treated as *defaults*: they only fill missing keys
        so that a project can still override the style in its own JSON.
        """
        for name in self._text_style_param_names:
            if name in config:
                continue
            try:
                # We do not force a type here: QSettings returns the same type
                # that was originally stored (bool/int/float/str).
                if self._settings.contains(f"lyrics/text_style/{name}"):
                    config[name] = self._settings.value(f"lyrics/text_style/{name}")
            except Exception:
                pass

        # Backward compatibility: if a project only stored `font_size` (legacy),
        # initialize `font_size_rel` so the UI and exports can scale properly.
        if "font_size_rel" not in config and isinstance(config.get("font_size"), (int, float)):
            try:
                config["font_size_rel"] = float(config.get("font_size", 40)) / 1080.0
            except Exception:
                pass

    def _apply_runtime_font_scaling(self) -> None:
        """
        Update the runtime `font_size` in the active plugin config based on the
        current widget height and the user-defined `font_size_rel`.

        This does **not** persist anything to the project. It only helps plugins
        that still rely on `config["font_size"]` (pixel size) to render text.
        """
        if self._current_plugin is None:
            return
        cfg = getattr(self._current_plugin, "config", None)
        if not isinstance(cfg, dict):
            return

        rel = cfg.get("font_size_rel", None)
        if not isinstance(rel, (int, float)):
            return

        h = int(self._current_plugin.height())
        if h <= 0:
            return

        # Clamp to a sensible range to avoid accidental extremes.
        rel_f = float(rel)
        rel_f = max(0.005, min(0.30, rel_f))

        px = int(round(rel_f * float(h)))
        px = max(8, min(2000, px))
        cfg["font_size"] = px
# ------------------------------------------------------------------
    # Slots (UI callbacks)
    # ------------------------------------------------------------------
    def _on_plugin_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        plugin_id = self.plugin_combo.itemData(index)
        if not plugin_id:
            return

        self._current_plugin_id = plugin_id

        # Update plugin description in the header box
        info = self._manager.get_info(plugin_id)
        if info is not None and hasattr(self, "plugin_description_label"):
            self.plugin_description_label.setText(info.description or "")
        elif hasattr(self, "plugin_description_label"):
            self.plugin_description_label.setText("")

        if self._current_plugin is not None and self._cover_pixmap is not None:
            self._current_plugin.cover_pixmap = self._cover_pixmap

        # When switching plugin, restore its last-known parameters (if any),
        # otherwise fall back to the legacy flat fields or to defaults.
        params: Dict[str, Any] = {}

        # Prefer per-plugin cache
        if plugin_id in self._plugin_params_by_id:
            params = dict(self._plugin_params_by_id[plugin_id])
        # Backward compatibility: reuse old flat config if it matches
        elif self._project is not None:
            stored_id = getattr(self._project, "lyrics_visual_plugin_id", None)
            stored_params = getattr(self._project, "lyrics_visual_parameters", {}) or {}
            if stored_id == plugin_id:
                params = dict(stored_params)

        self._create_plugin_instance(plugin_id, initial_config=params)

        # Rebuild parameter UI with the same info object
        self._rebuild_parameter_controls(info)

        # Restore per-plugin routing if we have it
        routing = self._plugin_routing_by_id.get(plugin_id)
        if routing is not None:
            source_id = routing.get("audio_source", "main")
            self._audio_source_id = source_id
            self._apply_audio_source_to_combo(source_id)
            self._update_audio_envelope()
        else:
            # Keep current audio source but recompute envelope, just in case
            self._update_audio_envelope()

        self._save_to_project()

    def _on_rescan_plugins_clicked(self) -> None:
        self._manager.refresh()
        self._rebuild_plugin_combo()

        # Try to restore the currently selected plugin id if possible
        if self._current_plugin_id:
            self._select_plugin_in_combo(self._current_plugin_id)

        # Ensure we have a valid instance for whatever is selected now
        idx = self.plugin_combo.currentIndex()
        if idx >= 0:
            plugin_id = self.plugin_combo.itemData(idx)
            if plugin_id:
                self._create_plugin_instance(plugin_id, initial_config=None)
                info = self._manager.get_info(plugin_id)
                self._rebuild_parameter_controls(info)

    def _on_audio_source_changed(self, index: int) -> None:
        if index < 0:
            return
        source_id = self.audio_source_combo.itemData(index)
        if not source_id:
            source_id = "main"
        self._audio_source_id = str(source_id)

        # Keep routing per plugin in memory
        if self._current_plugin_id:
            self._plugin_routing_by_id.setdefault(self._current_plugin_id, {})
            self._plugin_routing_by_id[self._current_plugin_id]["audio_source"] = (
                self._audio_source_id
            )

        self._update_audio_envelope()
        self._save_to_project()

    def _on_play_preview_clicked(self) -> None:
        """
        Play the currently selected audio source in the shared QMediaPlayer.

        This uses the same player as the Projects tab, but sets its source
        to the chosen stem or main mix so that the lyrics preview is driven
        by the correct audio.
        """
        if self._project is None:
            QMessageBox.information(
                self,
                "No project selected",
                "Please select a project in the Projects tab first.",
            )
            return

        audio_path = self._resolve_audio_path_for_source(self._audio_source_id)
        if audio_path is None or not audio_path.is_file():
            QMessageBox.warning(
                self,
                "No audio",
                "No audio file is available for the selected source.",
            )
            return

        # Set the global player source to this file and start playback.
        url = QUrl.fromLocalFile(str(audio_path.resolve()))
        self.player.setSource(url)
        self.player.play()

        # Ensure we have a matching envelope for the current audio.
        self._update_audio_envelope()
