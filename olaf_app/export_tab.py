from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING

import sys
import shutil
import soundfile as sf
import numpy as np
import traceback
import json   

try:
    import librosa  # type: ignore[import]
except ImportError:  # pragma: no cover
    librosa = None

from PyQt6.QtCore import Qt, QRect, QEvent
from PyQt6.QtGui import QPixmap, QImage, QFont, QPainter
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QCheckBox,
    QSizePolicy,
    QSlider,
    QApplication,
)


from .project_manager import Project
from .cover_visualizations_manager import CoverVisualizationsManager
from .cover_visualization_api import FrameFeatures, BaseCoverEffect
from .visualizations_manager import VisualizationManager, VisualizationPluginInfo
from .lyrics_visualizations_manager import LyricsVisualizationsManager  # <-- NEW
from .lyrics_visualization_api import LyricsFrameContext, BaseLyricsVisualization  # <-- NEW

if TYPE_CHECKING:
    from .visualization_api import BaseVisualization

# ----------------------------------------------------------------------
# Simple envelope struct (simplified copy of cover_visualizations_tab)
# ----------------------------------------------------------------------


@dataclass
class AudioEnvelope:
    """
    Simple structure holding an RMS envelope along with its metadata.

    Attributes
    ----------
    rms:
        1D numpy array containing RMS values in [0, 1] at a fixed FPS.
    fps:
        Frames per second of the envelope (i.e. how many RMS samples
        per second).
    duration:
        Duration in seconds covered by the envelope.
    """

    rms: np.ndarray
    fps: int
    duration: float


# ======================================================================
# ExportTab
# ======================================================================


class ExportTab(QWidget):
    """
    Final 2D visual export tab.

    Scope:
      - let the user configure:
          * video resolution (width, height),
          * visual selection (2D cover chain or 3D plugin),
          * FPS,
          * crop focus (horizontal/vertical),
          * audio section (start, duration);
      - show a single-frame preview, using:
          * the chosen visual (2D/3D) at a reduced resolution;
      - export a full video:
          * frames = cover visual chain (2D) or 3D plugin driven by RMS
            envelope;
          * audio = main project audio;
          * encoding via ffmpeg (H.264 + AAC).

    Additional features:
      - video size is taken from "Video settings" (Width/Height);
      - if aspect ratio differs from the cover, we crop the cover
        to match the target ratio;
      - cropping window is centered by default, but can be shifted
        using two controls:
          * Horizontal focus  (-100 = left,  0 = center, 100 = right),
          * Vertical focus    (-100 = top,   0 = center, 100 = bottom).
    """

    # ------------------------------------------------------------------
    # Constructor / basic wiring
    # ------------------------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._project: Optional[Project] = None

        # 2D cover manager and effect cache
        self._cover_manager = CoverVisualizationsManager()
        self._cover_manager.discover_plugins()
        self._cover_effect_cache: Dict[str, BaseCoverEffect] = {}

        # 3D visualization manager and plugin cache
        self._visual_manager = VisualizationManager()
        self._visual_manager.discover_plugins()
        self._visual_instances: Dict[str, "BaseVisualization"] = {}
        self._visual_widgets: Dict[str, QWidget] = {}

        # Lyrics visualization manager and cache (overlay on top of 2D / 3D)
        self._lyrics_manager = LyricsVisualizationsManager()
        self._lyrics_manager.refresh()
        self._lyrics_instances: Dict[str, BaseLyricsVisualization] = {}
        self._lyrics_phrases: List[Dict[str, Any]] = []
        self._lyrics_words: List[Dict[str, Any]] = []

        # Cache for audio envelopes: (path, fps) -> AudioEnvelope
        self._audio_cache: Dict[Tuple[Path, int], AudioEnvelope] = {}

        # Export state
        self._export_cancelled: bool = False
        # If True when _export_cancelled is set, we stop generating frames
        # but still encode a partial video from the frames already on disk.
        self._export_keep_partial: bool = False

        self._build_ui()

        # --- Interactive framing state (mouse drag on preview) --------
        self._dragging_frame = False
        self._drag_mode: Optional[str] = None  # "move" or "resize_br"
        self._drag_start_canvas_pos: Tuple[float, float] = (0.0, 0.0)
        self._drag_start_frame_rect = QRect()

        # Preview → canvas mapping (updated on each _update_preview)
        self._preview_canvas_w = 0
        self._preview_canvas_h = 0
        self._preview_display_w = 0
        self._preview_display_h = 0
        self._preview_frame_rect_canvas = QRect()

        # Listen to mouse+resize events on the preview label
        self.preview_label.installEventFilter(self)

        # Avoid recursive preview updates (e.g. Resize events triggered by
        # setPixmap inside _update_preview).
        self._in_update_preview = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_project(self, project: Optional[Project]) -> None:
        """
        Attach a Project instance to this tab. The project provides:

          - main audio path,
          - cover path,
          - cover_visual_effects configuration for 2D,
          - visualizations (3D plugins + routing),
          - lyrics visual configuration (style + routing).

        This will:

          - reset cached envelopes,
          - (re)build the 2D/3D visual combo,
          - (re)build the lyrics overlay combo,
          - reload alignment (phrases/words),
          - reset preview.
        """
        self._project = project
        self._audio_cache.clear()
        self._cover_effect_cache.clear()
        self._visual_instances.clear()
        self._visual_widgets.clear()
        self._lyrics_instances.clear()
        self._lyrics_phrases = []
        self._lyrics_words = []

        # 2D / 3D visual selection
        self._populate_2d_visual_combo()

        # Lyrics overlay combo
        self._populate_lyrics_combo()

        # Load alignment (if any) from project/vocal_align
        self._load_lyrics_alignment()

        # Default lyrics plugin selection from project metadata
        if self._project is not None:
            pid = getattr(self._project, "lyrics_visual_plugin_id", None)
            if pid:
                index_to_select = 0
                for i in range(self.combo_lyrics_plugin.count()):
                    if self.combo_lyrics_plugin.itemData(i) == pid:
                        index_to_select = i
                        break
                self.combo_lyrics_plugin.setCurrentIndex(index_to_select)
                # Enable overlay if a plugin is configured
                self.chk_enable_lyrics_overlay.setChecked(index_to_select != 0)
            else:
                self.combo_lyrics_plugin.setCurrentIndex(0)
                self.chk_enable_lyrics_overlay.setChecked(False)
        else:
            # No project -> disable overlay
            self.combo_lyrics_plugin.setCurrentIndex(0)
            self.chk_enable_lyrics_overlay.setChecked(False)

        self._update_preview()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """
        Build the Export tab UI.

        Layout:
          - Left column:
              * top row: canvas settings + output framing (side by side)
              * below:   visual preview (2D / 3D)
          - Right column:
              * video settings
              * video duration
              * lyrics overlay
              * visual source
              * export controls
        """
        # Top-level layout: left column (canvas + framing + preview)
        # and right column (other settings).
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # ------------------------ LEFT COLUMN ---------------------------
        left_panel = QWidget(self)
        left_column = QVBoxLayout(left_panel)
        left_column.setContentsMargins(0, 0, 0, 0)
        left_column.setSpacing(2)

        # --- Canvas settings (internal render resolution) --------------
        canvas_group = QGroupBox("Canvas settings", left_panel)
        canvas_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        canvas_group.setMinimumHeight(110)
        canvas_layout = QFormLayout(canvas_group)

        self.spin_canvas_width = QSpinBox(canvas_group)
        self.spin_canvas_width.setRange(320, 7680)
        self.spin_canvas_width.setSingleStep(16)
        # Default internal canvas = 1280x720
        self.spin_canvas_width.setValue(1280)

        self.slider_canvas_width = QSlider(Qt.Orientation.Horizontal, canvas_group)
        # Slider limited to a comfortable range; text box can go higher
        self.slider_canvas_width.setRange(320, 2560)

        self.slider_canvas_width.setSingleStep(16)
        self.slider_canvas_width.setValue(self.spin_canvas_width.value())

        width_row_widget = QWidget(canvas_group)
        width_row_layout = QHBoxLayout(width_row_widget)
        width_row_layout.setContentsMargins(0, 0, 0, 0)
        width_row_layout.setSpacing(4)
        width_row_layout.addWidget(self.spin_canvas_width)
        width_row_layout.addWidget(self.slider_canvas_width)

        self.spin_canvas_height = QSpinBox(canvas_group)
        self.spin_canvas_height.setRange(180, 4320)
        self.spin_canvas_height.setSingleStep(16)
        self.spin_canvas_height.setValue(720)

        self.slider_canvas_height = QSlider(Qt.Orientation.Horizontal, canvas_group)
        # Slider limité à 1440p en hauteur
        self.slider_canvas_height.setRange(180, 1440)

        self.slider_canvas_height.setSingleStep(16)
        self.slider_canvas_height.setValue(self.spin_canvas_height.value())

        height_row_widget = QWidget(canvas_group)
        height_row_layout = QHBoxLayout(height_row_widget)
        height_row_layout.setContentsMargins(0, 0, 0, 0)
        height_row_layout.setSpacing(4)
        height_row_layout.addWidget(self.spin_canvas_height)
        height_row_layout.addWidget(self.slider_canvas_height)

        canvas_layout.addRow("Canvas width:", width_row_widget)
        canvas_layout.addRow("Canvas height:", height_row_widget)

        # --- Output framing (canvas -> output) -------------------------
        frame_group = QGroupBox("Output framing (canvas crop)", left_panel)
        frame_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        frame_group.setMinimumHeight(110)
        frame_layout = QFormLayout(frame_group)

        # Reduce bottom padding to avoid unnecessary vertical scrollbars
        frame_layout.setContentsMargins(0, 0, 0, 2)
        frame_layout.setSpacing(2)

        self.chk_use_custom_frame = QCheckBox(
            "Use custom framing rectangle",
            frame_group,
        )
        self.chk_use_custom_frame.setChecked(True)
        frame_layout.addRow(self.chk_use_custom_frame)

        self.spin_frame_x = QSpinBox(frame_group)
        self.spin_frame_y = QSpinBox(frame_group)
        self.spin_frame_w = QSpinBox(frame_group)
        self.spin_frame_h = QSpinBox(frame_group)

        for sb in (
            self.spin_frame_x,
            self.spin_frame_y,
            self.spin_frame_w,
            self.spin_frame_h,
        ):
            sb.setRange(0, 10000)

        # sliders pour X et Y
        self.slider_frame_x = QSlider(Qt.Orientation.Horizontal, frame_group)
        self.slider_frame_x.setRange(0, 10000)
        self.slider_frame_x.setSingleStep(10)
        self.slider_frame_x.setValue(self.spin_frame_x.value())

        self.slider_frame_y = QSlider(Qt.Orientation.Horizontal, frame_group)
        self.slider_frame_y.setRange(0, 10000)
        self.slider_frame_y.setSingleStep(10)
        self.slider_frame_y.setValue(self.spin_frame_y.value())

        frame_x_row = QWidget(frame_group)
        frame_x_layout = QHBoxLayout(frame_x_row)
        frame_x_layout.setContentsMargins(0, 0, 0, 0)
        frame_x_layout.setSpacing(4)
        frame_x_layout.addWidget(self.spin_frame_x)
        frame_x_layout.addWidget(self.slider_frame_x)

        frame_y_row = QWidget(frame_group)
        frame_y_layout = QHBoxLayout(frame_y_row)
        frame_y_layout.setContentsMargins(0, 0, 0, 0)
        frame_y_layout.setSpacing(4)
        frame_y_layout.addWidget(self.spin_frame_y)
        frame_y_layout.addWidget(self.slider_frame_y)

        frame_layout.addRow("X (canvas px):", frame_x_row)
        frame_layout.addRow("Y (canvas px):", frame_y_row)
        frame_layout.addRow("Width (px):", self.spin_frame_w)
        frame_layout.addRow("Height (px):", self.spin_frame_h)

        # Default framing: full 1280x720 (matches default canvas/output)
        self.spin_frame_x.setValue(0)
        self.spin_frame_y.setValue(0)
        self.spin_frame_w.setValue(1280)
        self.spin_frame_h.setValue(720)
        self.slider_frame_x.setValue(0)
        self.slider_frame_y.setValue(0)

        self.btn_center_frame = QPushButton("Center frame in canvas", frame_group)
        frame_layout.addRow(self.btn_center_frame)

        # Row: [ Canvas settings | Output framing ]
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)
        top_row.addWidget(canvas_group, 1)
        top_row.addWidget(frame_group, 1)

        left_column.addLayout(top_row)

        # --- Visual preview (full width in left column) ----------------
        preview_group = QGroupBox("Visual preview (2D / 3D)", left_panel)
        preview_group.setFlat(False)
        preview_group.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        preview_vlayout = QVBoxLayout(preview_group)
        preview_vlayout.setContentsMargins(0, 0, 0, 0)
        preview_vlayout.setSpacing(4)

        self.preview_label = QLabel(
            "Preview will appear here.",
            preview_group,
        )
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 180)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.preview_label.setStyleSheet(
            "border: 1px solid #555; background-color: #111; color: #888;"
        )

        preview_vlayout.addWidget(self.preview_label)

        # Low-res hint
        self.preview_hint_label = QLabel(
            "Low-resolution preview only. Final export uses full video "
            "resolution and quality.",
            preview_group,
        )
        self.preview_hint_label.setWordWrap(True)
        self.preview_hint_label.setStyleSheet("color: #888888; font-size: 10px;")
        preview_vlayout.addWidget(self.preview_hint_label)

        left_column.addWidget(preview_group, 1)

        # Top row = paramètres compacts, preview = prend le reste en hauteur
        left_column.setStretch(0, 0)  # top_row
        left_column.setStretch(1, 1)  # preview_group

        # Add left column to root layout
        root_layout.addWidget(left_panel, stretch=3)

        # ------------------------ RIGHT COLUMN -------------------------
        right_column = QVBoxLayout()
        right_column.setContentsMargins(0, 0, 0, 0)
        right_column.setSpacing(8)

        # --- Output video settings (final encoded resolution) ----------
        video_group = QGroupBox("Video settings", self)
        video_layout = QFormLayout(video_group)

        # Preset for common output formats (16:9, 9:16, 1:1, etc.)
        self.combo_output_preset = QComboBox(video_group)
        self.combo_output_preset.addItem("Custom")  # index 0
        self.combo_output_preset.addItem("Match canvas")  # index 1
        self.combo_output_preset.addItem("16:9 landscape – 854×480 (480p)")   # 2
        self.combo_output_preset.addItem("16:9 landscape – 1280×720 (720p)")  # 3
        self.combo_output_preset.addItem("16:9 landscape – 1920×1080 (1080p)")  # 4
        self.combo_output_preset.addItem("16:9 landscape – 2560×1440 (1440p)")  # 5
        self.combo_output_preset.addItem("16:9 landscape – 3840×2160 (4K)")   # 6
        self.combo_output_preset.addItem("9:16 portrait – 720×1280")          # 7
        self.combo_output_preset.addItem("9:16 portrait – 1080×1920")         # 8
        self.combo_output_preset.addItem("1:1 square – 720×720")              # 9
        self.combo_output_preset.addItem("1:1 square – 1080×1080")            # 10

        video_layout.addRow("Output preset:", self.combo_output_preset)

        self.spin_width = QSpinBox(video_group)
        self.spin_width.setRange(320, 7680)
        self.spin_width.setSingleStep(16)
        self.spin_width.setValue(1280)

        self.spin_height = QSpinBox(video_group)
        self.spin_height.setRange(180, 4320)
        self.spin_height.setSingleStep(16)
        self.spin_height.setValue(720)

        self.spin_fps = QSpinBox(video_group)
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setValue(30)

        video_layout.addRow("Output width:", self.spin_width)
        video_layout.addRow("Output height:", self.spin_height)
        video_layout.addRow("FPS:", self.spin_fps)

        right_column.addWidget(video_group)

        # --- Video duration -------------------------------------------
        duration_group = QGroupBox("Video duration", self)
        duration_layout = QFormLayout(duration_group)

        self.spin_start_sec = QSpinBox(duration_group)
        self.spin_start_sec.setRange(0, 36000)
        self.spin_start_sec.setValue(0)
        duration_layout.addRow("Start (s):", self.spin_start_sec)

        self.spin_duration_sec = QSpinBox(duration_group)
        self.spin_duration_sec.setRange(1, 36000)
        self.spin_duration_sec.setValue(30)

        duration_row_widget = QWidget(duration_group)
        duration_row_layout = QHBoxLayout(duration_row_widget)
        duration_row_layout.setContentsMargins(0, 0, 0, 0)
        duration_row_layout.setSpacing(5)
        duration_row_layout.addWidget(self.spin_duration_sec)

        self.btn_full_song = QPushButton("Full song", duration_group)
        self.btn_full_song.setToolTip(
            "Set duration so the video covers the full song from the current start."
        )
        duration_row_layout.addWidget(self.btn_full_song)

        duration_layout.addRow("Duration (s):", duration_row_widget)

        right_column.addWidget(duration_group)

        # --- Lyrics overlay -------------------------------------------
        lyrics_group = QGroupBox("Lyrics overlay", self)
        lyrics_layout = QFormLayout(lyrics_group)

        self.chk_enable_lyrics_overlay = QCheckBox(
            "Enable lyrics overlay",
            lyrics_group,
        )
        lyrics_layout.addRow(self.chk_enable_lyrics_overlay)

        self.combo_lyrics_plugin = QComboBox(lyrics_group)
        lyrics_layout.addRow("Lyrics style:", self.combo_lyrics_plugin)

        self.lbl_lyrics_hint = QLabel(
            "Uses vocal alignment (phrases/words) and the text style configured "
            "in the Lyrics tab.",
            lyrics_group,
        )
        self.lbl_lyrics_hint.setWordWrap(True)
        self.lbl_lyrics_hint.setStyleSheet("color: #888888; font-size: 10px;")
        lyrics_layout.addRow(self.lbl_lyrics_hint)

        right_column.addWidget(lyrics_group)

        # --- Visual selection (2D / 3D) --------------------------------
        src_group = QGroupBox("Visual source (2D / 3D)", self)
        src_layout = QFormLayout(src_group)

        self.combo_visual_2d = QComboBox(src_group)
        self.combo_visual_2d.setEnabled(False)
        src_layout.addRow("Visual to export:", self.combo_visual_2d)

        right_column.addWidget(src_group)

        # --- Export controls group ------------------------------------
        export_group = QGroupBox("Export", self)
        export_layout = QVBoxLayout(export_group)

        buttons_row = QHBoxLayout()
        self.btn_export = QPushButton(
            "Render video (2D cover or 3D visualization)",
            export_group,
        )
        self.btn_stop_export = QPushButton("Stop", export_group)
        self.btn_stop_and_encode = QPushButton("Stop && encode partial", export_group)

        self.btn_stop_export.setEnabled(False)
        self.btn_stop_and_encode.setEnabled(False)

        buttons_row.addWidget(self.btn_export)
        buttons_row.addWidget(self.btn_stop_export)
        buttons_row.addWidget(self.btn_stop_and_encode)

        self.progress_export = QLabel("Idle", export_group)
        self.progress_export.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.progress_export.setStyleSheet("color: #888;")

        export_layout.addLayout(buttons_row)
        export_layout.addWidget(self.progress_export)

        right_column.addWidget(export_group)

        right_column.addStretch(1)
        root_layout.addLayout(right_column, stretch=2)

        # --- Signals ---------------------------------------------------
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_full_song.clicked.connect(self._on_full_song_clicked)
        self.btn_stop_export.clicked.connect(self._on_stop_export_clicked)
        self.btn_stop_and_encode.clicked.connect(self._on_stop_and_encode_clicked)

        self.spin_canvas_width.valueChanged.connect(self._on_canvas_width_changed)
        self.slider_canvas_width.valueChanged.connect(
            self._on_canvas_width_slider_changed
        )

        self.spin_canvas_height.valueChanged.connect(self._on_canvas_height_changed)
        self.slider_canvas_height.valueChanged.connect(
            self._on_canvas_height_slider_changed
        )

        self.spin_width.valueChanged.connect(self._update_preview)
        self.spin_height.valueChanged.connect(self._update_preview)
        self.spin_fps.valueChanged.connect(self._update_preview)
        self.combo_visual_2d.currentIndexChanged.connect(self._update_preview)
        self.chk_enable_lyrics_overlay.toggled.connect(self._update_preview)
        self.combo_lyrics_plugin.currentIndexChanged.connect(self._update_preview)

        self.combo_output_preset.currentIndexChanged.connect(
            self._on_output_preset_changed
        )

        self.chk_use_custom_frame.toggled.connect(self._update_preview)
        self.spin_frame_x.valueChanged.connect(self._on_frame_x_changed)
        self.slider_frame_x.valueChanged.connect(self._on_frame_x_slider_changed)
        self.spin_frame_y.valueChanged.connect(self._on_frame_y_changed)
        self.slider_frame_y.valueChanged.connect(self._on_frame_y_slider_changed)
        self.spin_frame_w.valueChanged.connect(self._update_preview)
        self.spin_frame_h.valueChanged.connect(self._update_preview)
        self.btn_center_frame.clicked.connect(self._on_center_frame_clicked)
        
        # Initialize frame sliders ranges to match the default canvas size
        self._update_frame_slider_ranges()
        

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_canvas_width_changed(self, value: int) -> None:
        ...
        if hasattr(self, "slider_canvas_width"):
            self.slider_canvas_width.blockSignals(True)
            self.slider_canvas_width.setValue(value)
            self.slider_canvas_width.blockSignals(False)

        self._update_frame_slider_ranges()
        self._update_preview()

    def _on_canvas_width_slider_changed(self, value: int) -> None:
        ...
        if hasattr(self, "spin_canvas_width"):
            self.spin_canvas_width.blockSignals(True)
            self.spin_canvas_width.setValue(value)
            self.spin_canvas_width.blockSignals(False)

        self._update_frame_slider_ranges()
        self._update_preview()

    def _on_canvas_height_changed(self, value: int) -> None:
        ...
        if hasattr(self, "slider_canvas_height"):
            self.slider_canvas_height.blockSignals(True)
            self.slider_canvas_height.setValue(value)
            self.slider_canvas_height.blockSignals(False)

        self._update_frame_slider_ranges()
        self._update_preview()

    def _on_canvas_height_slider_changed(self, value: int) -> None:
        ...
        if hasattr(self, "spin_canvas_height"):
            self.spin_canvas_height.blockSignals(True)
            self.spin_canvas_height.setValue(value)
            self.spin_canvas_height.blockSignals(False)

        self._update_frame_slider_ranges()
        self._update_preview()


    def _on_frame_x_changed(self, value: int) -> None:
        """
        Keep the frame X slider in sync with the spin box and refresh preview.
        """
        if hasattr(self, "slider_frame_x"):
            self.slider_frame_x.blockSignals(True)
            self.slider_frame_x.setValue(value)
            self.slider_frame_x.blockSignals(False)
        self._update_preview()

    def _on_frame_x_slider_changed(self, value: int) -> None:
        """
        Keep the frame X spin box in sync with the slider and refresh preview.
        """
        if hasattr(self, "spin_frame_x"):
            self.spin_frame_x.blockSignals(True)
            self.spin_frame_x.setValue(value)
            self.spin_frame_x.blockSignals(False)
        self._update_preview()

    def _on_frame_y_changed(self, value: int) -> None:
        """
        Keep the frame Y slider in sync with the spin box and refresh preview.
        """
        if hasattr(self, "slider_frame_y"):
            self.slider_frame_y.blockSignals(True)
            self.slider_frame_y.setValue(value)
            self.slider_frame_y.blockSignals(False)
        self._update_preview()

    def _on_frame_y_slider_changed(self, value: int) -> None:
        """
        Keep the frame Y spin box in sync with the slider and refresh preview.
        """
        if hasattr(self, "spin_frame_y"):
            self.spin_frame_y.blockSignals(True)
            self.spin_frame_y.setValue(value)
            self.spin_frame_y.blockSignals(False)
        self._update_preview()

    def _preview_size_from_video_settings(self) -> Tuple[int, int]:
        """
        Compute a reasonable preview size while preserving the aspect
        ratio of the **canvas** (internal render resolution).

        The preview itself reste en basse définition pour rester rapide.
        """
        target_w, target_h = self._get_canvas_size()

        max_w = 640
        aspect = target_w / max(1, target_h)

        if target_w <= max_w:
            return target_w, target_h

        preview_w = max_w
        preview_h = int(round(preview_w / aspect))
        return preview_w, preview_h

    def _get_canvas_size(self) -> Tuple[int, int]:
        """
        Return the internal canvas resolution used for rendering visuals.

        For now, this is configured via spin_canvas_width/height, with a
        safe fallback on output width/height if needed.
        """
        if hasattr(self, "spin_canvas_width") and hasattr(self, "spin_canvas_height"):
            w = max(1, int(self.spin_canvas_width.value()))
            h = max(1, int(self.spin_canvas_height.value()))
        else:
            # Fallback: use output settings
            w = max(1, int(self.spin_width.value()))
            h = max(1, int(self.spin_height.value()))
        return w, h

    def _update_frame_slider_ranges(self) -> None:
        """
        Keep frame X/Y sliders consistent with the current canvas size.
        This avoids 'violent' jumps when dragging or using the sliders.
        """
        canvas_w, canvas_h = self._get_canvas_size()
        if hasattr(self, "slider_frame_x"):
            self.slider_frame_x.blockSignals(True)
            self.slider_frame_x.setRange(0, max(0, canvas_w))
            self.slider_frame_x.blockSignals(False)
        if hasattr(self, "slider_frame_y"):
            self.slider_frame_y.blockSignals(True)
            self.slider_frame_y.setRange(0, max(0, canvas_h))
            self.slider_frame_y.blockSignals(False)

    def _get_output_size(self) -> Tuple[int, int]:
        """
        Return the final output video resolution (what is encoded to disk).
        """
        w = max(1, int(self.spin_width.value()))
        h = max(1, int(self.spin_height.value()))
        return w, h

    def _on_output_preset_changed(self) -> None:
        """
        Apply a common output format preset.

        This updates spin_width/spin_height. The user is still free to
        tweak the values afterwards (preset stays on the selected line,
        but acts as a one-shot helper).
        """
        if not hasattr(self, "combo_output_preset"):
            return

        index = self.combo_output_preset.currentIndex()

        # 0 = Custom -> do nothing
        if index == 0:
            return

        # Canvas size, used by "Match canvas"
        canvas_w, canvas_h = self._get_canvas_size()

        if index == 1:
            # Match canvas
            self.spin_width.setValue(canvas_w)
            self.spin_height.setValue(canvas_h)
        elif index == 2:
            # 16:9 landscape – 854x480 (480p)
            self.spin_width.setValue(854)
            self.spin_height.setValue(480)
        elif index == 3:
            # 16:9 landscape – 1280x720 (720p)
            self.spin_width.setValue(1280)
            self.spin_height.setValue(720)
        elif index == 4:
            # 16:9 landscape – 1920x1080 (1080p)
            self.spin_width.setValue(1920)
            self.spin_height.setValue(1080)
        elif index == 5:
            # 16:9 landscape – 2560x1440 (1440p)
            self.spin_width.setValue(2560)
            self.spin_height.setValue(1440)
        elif index == 6:
            # 16:9 landscape – 3840x2160 (4K)
            self.spin_width.setValue(3840)
            self.spin_height.setValue(2160)
        elif index == 7:
            # 9:16 portrait – 720x1280
            self.spin_width.setValue(720)
            self.spin_height.setValue(1280)
        elif index == 8:
            # 9:16 portrait – 1080x1920
            self.spin_width.setValue(1080)
            self.spin_height.setValue(1920)
        elif index == 9:
            # 1:1 square – 720x720
            self.spin_width.setValue(720)
            self.spin_height.setValue(720)
        elif index == 10:
            # 1:1 square – 1080x1080
            self.spin_width.setValue(1080)
            self.spin_height.setValue(1080)

        # The spinboxes are already connected to _update_preview, so the
        # preview will refresh automatically.

    def _compute_crop_rect(
        self,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
    ) -> QRect:
        """
        Compute a centered cropping rectangle for the cover image so that
        its aspect ratio matches the target aspect ratio.
        """
        if src_w <= 0 or src_h <= 0:
            return QRect(0, 0, src_w, src_h)

        video_aspect = target_w / max(1, target_h)
        src_aspect = src_w / max(1, src_h)

        if src_aspect > video_aspect:
            # Crop horizontally, keep full height
            new_w = int(round(src_h * video_aspect))
            new_h = src_h
            left = (src_w - new_w) // 2
            top = 0
        else:
            # Crop vertically, keep full width
            new_w = src_w
            new_h = int(round(src_w / video_aspect))
            left = 0
            top = (src_h - new_h) // 2

        return QRect(left, top, new_w, new_h)

    def _compute_auto_output_frame_rect(
        self,
        canvas_w: int,
        canvas_h: int,
        out_w: int,
        out_h: int,
    ) -> QRect:
        """
        Default framing: centered rectangle inside the canvas whose aspect
        ratio matches the final output aspect ratio.
        """
        if canvas_w <= 0 or canvas_h <= 0:
            return QRect(0, 0, canvas_w, canvas_h)

        out_aspect = out_w / max(1, out_h)
        canvas_aspect = canvas_w / max(1, canvas_h)

        if canvas_aspect > out_aspect:
            # Canvas trop large -> on coupe à gauche/droite
            new_w = int(round(canvas_h * out_aspect))
            new_h = canvas_h
            left = (canvas_w - new_w) // 2
            top = 0
        else:
            # Canvas trop haut -> on coupe en haut/en bas
            new_w = canvas_w
            new_h = int(round(canvas_w / out_aspect))
            left = 0
            top = (canvas_h - new_h) // 2

        return QRect(left, top, new_w, new_h)

    def _compute_output_frame_rect(
        self,
        canvas_w: int,
        canvas_h: int,
        out_w: int,
        out_h: int,
    ) -> QRect:
        """
        Compute the framing rectangle in canvas coordinates.

        If a custom rectangle is enabled, use the numeric X/Y/W/H values
        (clamped to the canvas). Otherwise, fall back to the centered
        rectangle and keep the spinboxes in sync.
        """
        if canvas_w <= 0 or canvas_h <= 0:
            return QRect(0, 0, canvas_w, canvas_h)

        use_custom = (
            hasattr(self, "chk_use_custom_frame")
            and self.chk_use_custom_frame.isChecked()
        )

        have_spins = all(
            hasattr(self, name)
            for name in (
                "spin_frame_x",
                "spin_frame_y",
                "spin_frame_w",
                "spin_frame_h",
            )
        )

        if use_custom and have_spins:
            x = max(0, int(self.spin_frame_x.value()))
            y = max(0, int(self.spin_frame_y.value()))
            w = max(1, int(self.spin_frame_w.value()))
            h = max(1, int(self.spin_frame_h.value()))

            # Clamp so the rectangle stays fully inside the canvas
            if x + w > canvas_w:
                w = max(1, canvas_w - x)
            if y + h > canvas_h:
                h = max(1, canvas_h - y)

            return QRect(x, y, w, h)

        # Fallback: automatic centered rectangle
        rect = self._compute_auto_output_frame_rect(canvas_w, canvas_h, out_w, out_h)

        # Keep numeric controls in sync (without déclencher _update_preview en boucle)
        if have_spins:
            updates = (
                (self.spin_frame_x, rect.x()),
                (self.spin_frame_y, rect.y()),
                (self.spin_frame_w, rect.width()),
                (self.spin_frame_h, rect.height()),
            )
            for sb, val in updates:
                sb.blockSignals(True)
                sb.setValue(val)
                sb.blockSignals(False)

        return rect

    def _extract_output_frame_from_canvas(
        self,
        canvas_img: QImage,
        frame_rect: QRect,
        out_w: int,
        out_h: int,
    ) -> QImage:
        """
        Given a full canvas image, crop it to the *fixed* framing rectangle
        and rescale to the final output resolution.

        The rectangle and output size are captured at export start so that
        changes in the UI do not affect a running export.
        """
        canvas_w = canvas_img.width()
        canvas_h = canvas_img.height()

        if canvas_w <= 0 or canvas_h <= 0:
            img = QImage(out_w, out_h, QImage.Format.Format_RGB32)
            img.fill(Qt.GlobalColor.black)
            return img

        rect = QRect(frame_rect)
        if rect.width() <= 0 or rect.height() <= 0:
            rect = self._compute_auto_output_frame_rect(canvas_w, canvas_h, out_w, out_h)

        # Clamp to canvas bounds
        if rect.x() < 0:
            rect.setX(0)
        if rect.y() < 0:
            rect.setY(0)
        if rect.right() >= canvas_w:
            rect.setRight(canvas_w - 1)
        if rect.bottom() >= canvas_h:
            rect.setBottom(canvas_h - 1)

        if rect.width() <= 0 or rect.height() <= 0:
            rect = QRect(0, 0, canvas_w, canvas_h)

        cropped = canvas_img.copy(rect)

        if cropped.width() != out_w or cropped.height() != out_h:
            cropped = cropped.scaled(
                out_w,
                out_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        return cropped

    def _on_center_frame_clicked(self) -> None:
        """
        Center the current framing rectangle inside the canvas.

        Width/height are kept as-is (clamped to the canvas size). Only
        the X/Y position is recomputed so that the rectangle is centered.
        """
        canvas_w, canvas_h = self._get_canvas_size()
        if canvas_w <= 0 or canvas_h <= 0:
            return

        if not all(
            hasattr(self, name)
            for name in (
                "spin_frame_x",
                "spin_frame_y",
                "spin_frame_w",
                "spin_frame_h",
                "chk_use_custom_frame",
            )
        ):
            return

        # Use current frame width/height, with a safe clamp
        cur_w = int(self.spin_frame_w.value())
        cur_h = int(self.spin_frame_h.value())

        if cur_w <= 0 or cur_h <= 0:
            # If not initialized, fall back to an automatic rect
            out_w, out_h = self._get_output_size()
            auto_rect = self._compute_auto_output_frame_rect(
                canvas_w, canvas_h, out_w, out_h
            )
            cur_w = auto_rect.width()
            cur_h = auto_rect.height()

        cur_w = max(1, min(cur_w, canvas_w))
        cur_h = max(1, min(cur_h, canvas_h))

        left = (canvas_w - cur_w) // 2
        top = (canvas_h - cur_h) // 2

        updates = (
            (self.spin_frame_x, left),
            (self.spin_frame_y, top),
            (self.spin_frame_w, cur_w),
            (self.spin_frame_h, cur_h),
        )
        for sb, val in updates:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

        self.chk_use_custom_frame.setChecked(True)
        self._update_preview()

    # ------------------------------------------------------------------
    # Visual combo population (2D + 3D)
    # ------------------------------------------------------------------
    def _populate_2d_visual_combo(self) -> None:
        """
        Fill the combo with available visuals for this project:
        - 2D cover effect chain (if configured);
        - 3D visualization plugins saved in Project.visualizations.

        The combo userData encodes the selection as:
            "2d:cover_chain"  or  "3d:<plugin_id>".
        """
        self.combo_visual_2d.clear()

        if self._project is None:
            self.combo_visual_2d.setEnabled(False)
            self.combo_visual_2d.addItem("No project loaded", userData=None)
            return

        has_any = False

        # --- 2D cover chain ------------------------------------------------
        chain = getattr(self._project, "cover_visual_chain", None) or []
        if chain:
            self.combo_visual_2d.setEnabled(True)
            self.combo_visual_2d.addItem(
                "[2D] Project cover effect chain",
                userData="2d:cover_chain",
            )
            has_any = True

        # --- 3D visualizations saved in project ----------------------------
        visualizations: Dict[str, Dict[str, Any]] = getattr(
            self._project, "visualizations", {}
        ) or {}

        if visualizations:
            # Make sure we have an up-to-date plugin list to resolve names.
            self._visual_manager.discover_plugins()
            for plugin_id in sorted(visualizations.keys()):
                info = self._visual_manager.get_plugin(plugin_id)
                name = info.name if info is not None else plugin_id
                label = f"[3D] {name}"
                user_data = f"3d:{plugin_id}"
                self.combo_visual_2d.addItem(label, userData=user_data)
                has_any = True

        if not has_any:
            self.combo_visual_2d.setEnabled(False)
            self.combo_visual_2d.addItem("No 2D/3D visual in project", userData=None)
            return

        # Try to restore last-used 3D plugin if present, otherwise 2D chain
        selected_key: Optional[str] = None
        last_3d = getattr(self._project, "visualization_plugin_id", None)
        if last_3d and last_3d in visualizations:
            selected_key = f"3d:{last_3d}"
        elif chain:
            selected_key = "2d:cover_chain"

        if selected_key is not None:
            idx = self.combo_visual_2d.findData(selected_key)
            if idx >= 0:
                self.combo_visual_2d.setCurrentIndex(idx)

    def _populate_lyrics_combo(self) -> None:
        """
        Populate the lyrics overlay combo from the lyrics visualization manager.

        The first entry is always "No lyrics overlay" (userData=None).
        """
        if not hasattr(self, "combo_lyrics_plugin"):
            return

        self.combo_lyrics_plugin.blockSignals(True)
        self.combo_lyrics_plugin.clear()

        # First entry: disabled overlay
        self.combo_lyrics_plugin.addItem("No lyrics overlay", None)

        try:
            self._lyrics_manager.refresh()
            visuals = self._lyrics_manager.available_visuals()
        except Exception:
            visuals = []

        for info in visuals:
            self.combo_lyrics_plugin.addItem(info.name, info.plugin_id)

        self.combo_lyrics_plugin.blockSignals(False)


    # ------------------------------------------------------------------
    # Preview logic
    # ------------------------------------------------------------------
    def _update_preview(self) -> None:
        """
        Render a preview of the *canvas* (2D / 3D) and draw the current
        output framing rectangle on top.

        This preview is:
          - rendered at a reduced canvas resolution for speed,
          - NOT cropped to the output format,
          - WITHOUT lyrics overlay (lyrics are applied later in the
            export pipeline, after framing).

        The goal here is to make the framing easier to understand:
        you see the full canvas, and the rectangle that will be used
        to generate the final video frames.
        """
        if self._project is None:
            return

        # Guard against re-entrant calls (e.g. Resize inside setPixmap)
        if getattr(self, "_in_update_preview", False):
            return
        self._in_update_preview = True

        try:
            mode = self.combo_visual_2d.currentData()

            # Preview canvas size: same aspect ratio as the real canvas,
            # but smaller for performance.
            canvas_preview_w, canvas_preview_h = self._preview_size_from_video_settings()

            canvas_img: Optional[QImage] = None
            preview_t = 0.0
            fps = max(1, self.spin_fps.value())

            # 1) Render a single canvas frame (2D or 3D) -----------------
            if isinstance(mode, str) and mode.startswith("3d:"):
                # 3D visualization preview (static frame at t = 0)
                plugin_id = mode.split(":", 1)[1]
                canvas_img = self._render_3d_preview_frame(
                    plugin_id, canvas_preview_w, canvas_preview_h
                )
            else:
                # 2D cover chain preview
                chain = getattr(self._project, "cover_visual_chain", None) or []
                if not chain:
                    self.preview_label.setText(
                        "No 2D cover chain configured in this project.",
                    )
                    self.preview_label.setPixmap(QPixmap())
                    return

                canvas_img = self._render_cover_chain_frame(
                    canvas_preview_w,
                    canvas_preview_h,
                    t=preview_t,
                    amp=0.8,
                )

            if canvas_img is None:
                canvas_img = self._make_base_frame(canvas_preview_w, canvas_preview_h)

            # 2) Compute framing rectangle in *real* canvas coordinates ---
            true_canvas_w, true_canvas_h = self._get_canvas_size()
            out_w, out_h = self._get_output_size()

            rect_canvas = self._compute_output_frame_rect(
                true_canvas_w,
                true_canvas_h,
                out_w,
                out_h,
            )

            # Store mapping info for mouse interactions
            self._preview_canvas_w = true_canvas_w
            self._preview_canvas_h = true_canvas_h
            self._preview_frame_rect_canvas = QRect(rect_canvas)

            # Map the canvas rectangle to the preview image coordinates.
            if true_canvas_w <= 0 or true_canvas_h <= 0:
                scale_x = 1.0
                scale_y = 1.0
            else:
                scale_x = canvas_preview_w / float(true_canvas_w)
                scale_y = canvas_preview_h / float(true_canvas_h)

            frame_x = int(round(rect_canvas.x() * scale_x))
            frame_y = int(round(rect_canvas.y() * scale_y))
            frame_w = int(round(rect_canvas.width() * scale_x))
            frame_h = int(round(rect_canvas.height() * scale_y))

            frame_x = max(0, min(frame_x, canvas_preview_w - 1))
            frame_y = max(0, min(frame_y, canvas_preview_h - 1))
            frame_w = max(1, min(frame_w, canvas_preview_w - frame_x))
            frame_h = max(1, min(frame_h, canvas_preview_h - frame_y))

            # 3) Draw the framing rectangle on top of the canvas image ----
            img_with_frame = canvas_img.convertToFormat(QImage.Format.Format_RGB32)
            painter = QPainter(img_with_frame)
            try:
                painter.setPen(Qt.GlobalColor.white)
                painter.drawRect(frame_x, frame_y, frame_w - 1, frame_h - 1)

                # --- Draw visual handles (small squares) on the corners ---
                handle_size = 8
                half = handle_size // 2

                # Corner centers in preview coordinates
                corners = [
                    (frame_x, frame_y),  # top-left
                    (frame_x + frame_w - 1, frame_y),  # top-right
                    (frame_x, frame_y + frame_h - 1),  # bottom-left
                    (frame_x + frame_w - 1, frame_y + frame_h - 1),  # bottom-right
                ]

                painter.setBrush(Qt.GlobalColor.white)
                for cx, cy in corners:
                    painter.drawRect(
                        cx - half,
                        cy - half,
                        handle_size,
                        handle_size,
                    )
            finally:
                painter.end()


            # 4) Scale the preview image to the preview label -------------
            pix = QPixmap.fromImage(img_with_frame)

            target_w = self.preview_label.width()
            target_h = self.preview_label.height()

            # Premier appel: label pas encore dimensionné -> on ne rescale pas
            if target_w > 0 and target_h > 0:
                pix = pix.scaled(
                    target_w,
                    target_h,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )

            # Mémoriser la taille réelle de l'image affichée
            self._preview_display_w = pix.width()
            self._preview_display_h = pix.height()

            self.preview_label.setPixmap(pix)
            self.preview_label.setText("")
        except Exception as exc:  # pragma: no cover
            print("ExportTab preview error:", exc, file=sys.stderr)
        finally:
            # Toujours relâcher le flag, même en cas d’exception
            self._in_update_preview = False

    def _map_preview_pos_to_canvas(
        self,
        x: int,
        y: int,
    ) -> Optional[Tuple[float, float]]:
        """
        Map a mouse position in preview_label coordinates (pixels) to
        canvas coordinates (same space as the framing rectangle).

        Returns (cx, cy) in canvas coordinates, or None if the point is
        outside the displayed image area.
        """
        if (
            self._preview_canvas_w <= 0
            or self._preview_canvas_h <= 0
            or self._preview_display_w <= 0
            or self._preview_display_h <= 0
        ):
            return None

        label_w = self.preview_label.width()
        label_h = self.preview_label.height()

        # Image centrée dans le label (KeepAspectRatio)
        offset_x = (label_w - self._preview_display_w) // 2
        offset_y = (label_h - self._preview_display_h) // 2

        px = x - offset_x
        py = y - offset_y

        if px < 0 or py < 0 or px >= self._preview_display_w or py >= self._preview_display_h:
            return None

        # display -> canvas
        scale_x = self._preview_canvas_w / float(self._preview_display_w)
        scale_y = self._preview_canvas_h / float(self._preview_display_h)

        cx = px * scale_x
        cy = py * scale_y
        return cx, cy

    def _start_frame_drag(self, pos) -> None:
        """
        Start dragging / resizing the framing rectangle from a mouse press.
        """
        mapped = self._map_preview_pos_to_canvas(pos.x(), pos.y())
        if mapped is None:
            return

        cx, cy = mapped
        canvas_w, canvas_h = self._get_canvas_size()
        out_w, out_h = self._get_output_size()
        rect = self._compute_output_frame_rect(canvas_w, canvas_h, out_w, out_h)

        # Petite marge pour la détection du coin bas-droite (resize)
        # Utilisée en coordonnées canvas.
        if self._preview_display_w > 0:
            margin_canvas = 10.0 * canvas_w / float(self._preview_display_w)
        else:
            margin_canvas = 10.0

        br_x = rect.x() + rect.width()
        br_y = rect.y() + rect.height()

        near_br = (
            abs(cx - br_x) <= margin_canvas
            and abs(cy - br_y) <= margin_canvas
        )

        if near_br:
            mode = "resize_br"
        elif rect.contains(int(cx), int(cy)):
            mode = "move"
        else:
            # Click en dehors du cadre -> aucune interaction
            return

        self._dragging_frame = True
        self._drag_mode = mode
        self._drag_start_canvas_pos = (cx, cy)
        self._drag_start_frame_rect = QRect(rect)

        # S'assurer qu'on est bien en mode custom
        if hasattr(self, "chk_use_custom_frame"):
            self.chk_use_custom_frame.setChecked(True)

    def _update_frame_drag(self, pos) -> None:
        """
        Update framing rectangle while dragging.

        - If mode == "move": translate the rectangle inside the canvas.
        - If mode == "resize_br": resize from bottom-right corner while
          preserving the output aspect ratio (output width / output height).
        """
        if not self._dragging_frame or not self._drag_mode:
            return

        mapped = self._map_preview_pos_to_canvas(pos.x(), pos.y())
        if mapped is None:
            return

        cx, cy = mapped
        start_cx, start_cy = self._drag_start_canvas_pos
        rect0 = self._drag_start_frame_rect
        canvas_w, canvas_h = self._get_canvas_size()

        dx = cx - start_cx
        dy = cy - start_cy

        if self._drag_mode == "move":
            # Simple translation, no change in size
            new_x = int(round(rect0.x() + dx))
            new_y = int(round(rect0.y() + dy))
            new_x = max(0, min(new_x, canvas_w - rect0.width()))
            new_y = max(0, min(new_y, canvas_h - rect0.height()))
            new_rect = QRect(new_x, new_y, rect0.width(), rect0.height())

        elif self._drag_mode == "resize_br":
            # Resize from bottom-right corner, preserving output aspect ratio
            out_w, out_h = self._get_output_size()
            if out_w <= 0 or out_h <= 0:
                aspect = rect0.width() / max(1, rect0.height())
            else:
                aspect = out_w / float(out_h)

            # On utilise le déplacement horizontal comme driver principal
            base_new_w = rect0.width() + dx
            if base_new_w < 1:
                base_new_w = 1

            # Clamp largeur à ce qui reste dans le canvas
            max_w = max(1, canvas_w - rect0.x())
            base_new_w = min(base_new_w, max_w)

            # Hauteur dérivée du ratio
            base_new_h = int(round(base_new_w / aspect))
            if base_new_h < 1:
                base_new_h = 1

            # Si on déborde verticalement, on recalcule en se basant sur la hauteur
            if rect0.y() + base_new_h > canvas_h:
                max_h = max(1, canvas_h - rect0.y())
                base_new_h = max_h
                base_new_w = int(round(base_new_h * aspect))
                if rect0.x() + base_new_w > canvas_w:
                    base_new_w = max(1, canvas_w - rect0.x())

            new_w = max(1, int(base_new_w))
            new_h = max(1, int(base_new_h))

            new_rect = QRect(rect0.x(), rect0.y(), new_w, new_h)

        else:
            return

        # Répercuter dans les spinbox, sans boucles de signaux
        if all(
            hasattr(self, name)
            for name in ("spin_frame_x", "spin_frame_y", "spin_frame_w", "spin_frame_h")
        ):
            updates = (
                (self.spin_frame_x, new_rect.x()),
                (self.spin_frame_y, new_rect.y()),
                (self.spin_frame_w, new_rect.width()),
                (self.spin_frame_h, new_rect.height()),
            )
            for sb, val in updates:
                sb.blockSignals(True)
                sb.setValue(val)
                sb.blockSignals(False)

        self._update_preview()

    def _end_frame_drag(self) -> None:
        """
        End current frame drag / resize interaction.
        """
        self._dragging_frame = False
        self._drag_mode = None

    def eventFilter(self, obj, event) -> bool:
        """
        Handle mouse interaction on the preview label:

          - Mouse press: start moving/resizing the framing rectangle.
          - Mouse move: update rectangle.
          - Mouse release: stop interaction.
          - Resize: recompute preview to match new size.
        """
        if obj is self.preview_label:
            et = event.type()

            if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._start_frame_drag(event.pos())
                # On laisse QLabel gérer le reste, donc on renvoie False
                return False

            if et == QEvent.Type.MouseMove and self._dragging_frame:
                self._update_frame_drag(event.pos())
                return False

            if et == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self._end_frame_drag()
                return False

            if et == QEvent.Type.Resize:
                # Quand la preview change de taille, on recalcule l'image
                if self._project is not None:
                    self._update_preview()
                return False

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Lyrics alignment loading and context building
    # ------------------------------------------------------------------
    def _load_lyrics_alignment(self) -> None:
        """
        Load phrases/words alignment from the project folder, if available.

        Files:
          - project/vocal_align/phrases.json
          - project/vocal_align/words.json
        """
        self._lyrics_phrases = []
        self._lyrics_words = []

        if self._project is None:
            return

        align_dir = self._project.folder / "vocal_align"
        phrases_path = align_dir / "phrases.json"
        words_path = align_dir / "words.json"

        if not phrases_path.is_file():
            # No alignment -> overlay will silently show nothing.
            return

        try:
            phrases_data = json.loads(phrases_path.read_text(encoding="utf-8"))
            if isinstance(phrases_data, list):
                self._lyrics_phrases = phrases_data
        except Exception:
            self._lyrics_phrases = []
            return

        if words_path.is_file():
            try:
                words_data = json.loads(words_path.read_text(encoding="utf-8"))
                if isinstance(words_data, list):
                    self._lyrics_words = words_data
            except Exception:
                self._lyrics_words = []
        else:
            self._lyrics_words = []

    def _build_lyrics_context_for_time(self, t: float, amp: float) -> LyricsFrameContext:
        """
        Build a LyricsFrameContext at time t using phrases/words alignment.

        This mirrors the logic used in LyricsVisualizationsTab so that
        export and live preview share exactly the same behavior.
        """
        phrase_index: Optional[int] = None
        phrase_obj: Optional[Dict[str, Any]] = None
        phrase_start = 0.0
        phrase_end = 0.0

        # 1) Find active phrase
        for idx, phrase in enumerate(self._lyrics_phrases or []):
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
            # Outside of any phrase -> empty context
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

        # 2) Determine line_index for words
        line_index = phrase_obj.get("line_index", phrase_index)

        # All words for this line, in temporal order
        line_words: List[Tuple[int, Dict[str, Any]]] = []
        for gidx, w in enumerate(self._lyrics_words or []):
            if w.get("line_index", line_index) != line_index:
                continue
            line_words.append((gidx, w))

        # 3) Find active word by timing
        current_word_text: Optional[str] = None
        current_word_global_idx: Optional[int] = None

        for gidx, w in line_words:
            try:
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws))
            except Exception:
                continue
            if ws <= t <= we:
                current_word_text = str(w.get("text", ""))
                current_word_global_idx = gidx
                break

        # 4) Compute occurrence index of the active word inside the phrase
        word_char_start: Optional[int] = None
        word_char_end: Optional[int] = None

        if current_word_text and current_word_global_idx is not None:
            lower_line = text_full_line.lower()
            lower_word = current_word_text.lower()

            occurrence_index = 0
            for gidx, w in line_words:
                if gidx == current_word_global_idx:
                    break
                if str(w.get("text", "")).lower() == lower_word:
                    occurrence_index += 1

            pos = -1
            start_search = 0
            for _ in range(occurrence_index + 1):
                pos = lower_line.find(lower_word, start_search)
                if pos < 0:
                    break
                start_search = pos + len(lower_word)

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
    # Envelope helpers
    # ------------------------------------------------------------------
    def _ensure_audio_envelope(self, path: Path, fps: int) -> AudioEnvelope:
        """
        Compute (or reuse from cache) an RMS envelope of the given audio
        file at the specified FPS.
        """
        key = (path, fps)
        if key in self._audio_cache:
            return self._audio_cache[key]

        if not path.is_file():
            raise FileNotFoundError(str(path))

        if librosa is None:
            data, sr = sf.read(str(path), always_2d=True)
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            mono = data.mean(axis=1)
            hop_length = int(sr / fps)
            if hop_length <= 0:
                hop_length = 1

            n_frames = len(mono) // hop_length
            if n_frames <= 0:
                env = AudioEnvelope(
                    rms=np.zeros(1, dtype=np.float32),
                    fps=fps,
                    duration=0.0,
                )
                self._audio_cache[key] = env
                return env

            rms_values = []
            for i in range(n_frames):
                start = i * hop_length
                end = min(len(mono), start + hop_length)
                chunk = mono[start:end]
                if len(chunk) == 0:
                    rms = 0.0
                else:
                    rms = float(np.sqrt(np.mean(chunk * chunk)))
                rms_values.append(rms)

            rms_arr = np.array(rms_values, dtype=np.float32)
            rms_arr /= max(1e-6, rms_arr.max() or 1.0)
            duration = len(mono) / float(sr)
        else:
            y, sr = librosa.load(str(path), sr=None, mono=True)
            hop_length = int(sr / fps)
            if hop_length <= 0:
                hop_length = 1

            rms = librosa.feature.rms(y=y, frame_length=hop_length * 2, hop_length=hop_length)[0]
            if len(rms) == 0:
                env = AudioEnvelope(
                    rms=np.zeros(1, dtype=np.float32),
                    fps=fps,
                    duration=0.0,
                )
                self._audio_cache[key] = env
                return env

            rms_arr = rms.astype(np.float32)
            rms_arr /= max(1e-6, rms_arr.max() or 1.0)
            duration = len(y) / float(sr)

        env = AudioEnvelope(rms=rms_arr, fps=fps, duration=duration)
        self._audio_cache[key] = env
        return env

    def _sample_envelope(self, env: AudioEnvelope, t: float) -> float:
        """
        Sample the envelope at an arbitrary time in seconds.
        """
        if env.duration <= 0.0 or len(env.rms) == 0:
            return 0.0

        if t <= 0.0:
            return float(env.rms[0])
        if t >= env.duration:
            return float(env.rms[-1])

        index = int(t * env.fps)
        index = max(0, min(index, len(env.rms) - 1))
        return float(env.rms[index])

    def _resolve_cover_audio_source_path(self, source_id: str) -> Optional[Path]:
        """
        Resolve the audio file path for a 2D cover effect source_id.

        source_id can be:
          - 'main'       -> project's main audio file
          - 'stem:xxxx'  -> first matching stem named 'xxxx' in stems_by_model
        """
        if self._project is None:
            return None

        # Main mix
        if source_id == "main":
            try:
                return self._project.get_audio_path()
            except Exception:
                return None

        # Stems (search across all stem models)
        if source_id.startswith("stem:"):
            stem_name = source_id.split(":", 1)[1]
            stems_by_model = getattr(self._project, "stems_by_model", {}) or {}
            for _, stems in stems_by_model.items():
                rel_path = stems.get(stem_name)
                if rel_path:
                    return self._project.folder / rel_path

        return None

    # ------------------------------------------------------------------
    # 2D cover chain rendering
    # ------------------------------------------------------------------
    def _render_cover_chain_frame(
        self,
        width: int,
        height: int,
        t: float,
        amp: float,
        amp_by_entry: Optional[Dict[str, float]] = None,
    ) -> Optional[QImage]:
        """
        Render one frame of the 2D cover visual chain.

        Steps:
          - load the cover,
          - crop it to match the target aspect ratio (using focus sliders),
          - scale crop to (width, height),
          - convert to RGB numpy array,
          - run each effect in cover_visual_chain, using:
              * amp_by_entry[entry_key] if provided,
              * otherwise the global 'amp' value,
          - build a QImage from the final frame.
        """
        if self._project is None:
            return None

        # Older projects without cover_file should not try to render
        if not getattr(self._project, "cover_file", None):
            return None

        chain: List[str] = getattr(self._project, "cover_visual_chain", None) or []
        if not chain:
            return None

        effects_cfg: Dict[str, Dict[str, Any]] = getattr(
            self._project, "cover_visual_effects", None
        ) or {}
        if not effects_cfg:
            return None

        cover_path = self._project.get_cover_path()
        if cover_path is None or not cover_path.is_file():
            return None

        pix = QPixmap(str(cover_path))
        if pix.isNull():
            return None

        src_w = pix.width()
        src_h = pix.height()
        crop_rect = self._compute_crop_rect(src_w, src_h, width, height)
        cropped = pix.copy(crop_rect)

        # At this stage cropped has the correct aspect ratio; we can scale
        # directly to (width, height) without distorting.
        scaled = cropped.scaled(
            width,
            height,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Convert to tight RGB image -> numpy array (uint8, 0–255)
        image = scaled.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w = image.width()
        h = image.height()
        ptr = image.bits()
        ptr.setsize(h * w * 3)
        frame = np.frombuffer(ptr, np.uint8).reshape((h, w, 3)).copy()

        # Default global amplitude
        global_amp = float(amp)

        # Apply effects in order
        for entry_key in chain:
            cfg_entry = effects_cfg.get(entry_key)
            if not isinstance(cfg_entry, dict):
                continue

            effect_id = cfg_entry.get("effect_id")
            params = cfg_entry.get("parameters", {}) or {}
            if not effect_id:
                continue

            # Per-effect amplitude override
            entry_amp = global_amp
            if amp_by_entry is not None and entry_key in amp_by_entry:
                entry_amp = float(amp_by_entry.get(entry_key, global_amp))

            features = FrameFeatures(amp=float(entry_amp))

            # Reuse / create effect instance
            effect = self._cover_effect_cache.get(entry_key)
            if (
                not isinstance(effect, BaseCoverEffect)
                or getattr(effect, "effect_id", None) != effect_id
            ):
                effect = self._cover_manager.create_instance(effect_id, config=params)
                if effect is None:
                    continue
                self._cover_effect_cache[entry_key] = effect
            else:
                # Keep cache in sync with latest parameters
                try:
                    effect.load_state(params)
                except Exception:
                    pass

            # Apply effect – same API as in cover_visualizations_tab.py
            try:
                frame = effect.apply_to_frame(frame, t=t, features=features)
            except Exception:
                # Never crash export because of one effect
                continue

            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
            ):
                # Plugin returned something unexpected
                return None

        # Clamp and convert back to QImage (RGB32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out_h, out_w, _ = frame.shape

        out_image = QImage(
            frame.data,
            out_w,
            out_h,
            3 * out_w,  # bytes per line (RGB)
            QImage.Format.Format_RGB888,
        )
        return out_image.copy().convertToFormat(QImage.Format.Format_RGB32)

    # ------------------------------------------------------------------
    # 3D visualization helpers (off-screen rendering)
    # ------------------------------------------------------------------
    def _get_or_create_visualization_instance(
        self, plugin_id: str, width: int, height: int, fps: int
    ) -> Tuple["BaseVisualization", QWidget]:
        """
        Lazily create (or reuse) a 3D visualization instance and its
        off-screen widget for the given resolution and FPS.

        The widget is kept as a *child* of this tab (so the OpenGL context
        is valid) but its geometry is moved outside of the visible area so
        it never appears on screen. We still grab() its contents to build
        preview / export frames.

        IMPORTANT:
        Even when reusing an existing instance, we re-synchronize its
        configuration from the Project.visualizations[...] parameters so
        that changes made in the Visualizations tab are reflected in the
        export.
        """
        if self._project is None:
            raise RuntimeError("No project selected.")

        # Always fetch latest saved state from the project
        visualizations: Dict[str, Dict[str, Any]] = getattr(
            self._project, "visualizations", {}
        ) or {}
        state = visualizations.get(plugin_id, {})
        params = state.get("parameters") or {}

        # ------------------------------------------------------------------
        # Reuse existing instance + widget when possible
        # ------------------------------------------------------------------
        if plugin_id in self._visual_instances:
            instance = self._visual_instances[plugin_id]
            widget = self._visual_widgets.get(plugin_id)

            # Re-synchronize instance.config with latest project parameters
            try:
                cfg = getattr(instance, "config", {}) or {}
                cfg.update(params)
            except Exception:
                pass

            if widget is not None:
                # Keep a fixed off-screen geometry so it never overlaps the UI
                widget.setGeometry(-width - 50, -height - 50, width, height)
                widget.setMinimumSize(width, height)
                widget.setMaximumSize(width, height)

            try:
                instance.apply_preview_settings(width, height, fps)  # type: ignore[attr-defined]
            except Exception:
                pass

            if widget is None:
                raise RuntimeError("Visualization widget is missing.")
            return instance, widget

        # ------------------------------------------------------------------
        # Fresh instance from plugin metadata saved in the project
        # ------------------------------------------------------------------
        info = self._visual_manager.get_plugin(plugin_id)
        if info is None:
            raise RuntimeError(f"Unknown visualization plugin: {plugin_id}")

        instance = self._visual_manager.create_instance(plugin_id, config=None)
        if instance is None:
            raise RuntimeError(f"Could not create visualization instance: {plugin_id}")

        # Apply saved parameters into the instance config
        try:
            cfg = getattr(instance, "config", {}) or {}
            cfg.update(params)
        except Exception:
            pass

        # Off-screen widget: child of this tab, but moved outside visible area
        widget = instance.create_widget(self)  # type: ignore[attr-defined]
        widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        widget.setGeometry(-width - 50, -height - 50, width, height)
        widget.setMinimumSize(width, height)
        widget.setMaximumSize(width, height)

        try:
            instance.apply_preview_settings(width, height, fps)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            instance.on_activate()  # type: ignore[attr-defined]
        except Exception:
            pass

        self._visual_instances[plugin_id] = instance
        self._visual_widgets[plugin_id] = widget
        return instance, widget


    def _render_3d_preview_frame(
        self, plugin_id: str, width: int, height: int
    ) -> Optional[QImage]:
        """
        Render a single static preview frame for the given 3D plugin.

        We do not depend on real audio here; instead we send a synthetic
        feature vector with a moderate RMS so the plugin shows its look.
        """
        try:
            fps = max(1, int(self.spin_fps.value()))
            instance, widget = self._get_or_create_visualization_instance(
                plugin_id, width, height, fps
            )
        except Exception:
            return None

        visualizations: Dict[str, Dict[str, Any]] = getattr(
            self._project, "visualizations", {}
        ) or {}
        state = visualizations.get(plugin_id, {})
        routing = state.get("routing") or {}

        inputs: Dict[str, Dict[str, float]] = {}
        if routing:
            for input_key in routing.keys():
                inputs[input_key] = {"rms": 0.8}
        else:
            inputs["input_1"] = {"rms": 0.8}

        features = {"time_ms": 0, "inputs": inputs}
        try:
            instance.on_audio_features(features)  # type: ignore[attr-defined]
        except Exception:
            pass

        QApplication.processEvents()
        pix = widget.grab()
        if pix.isNull():
            return None

        img = pix.toImage().convertToFormat(QImage.Format.Format_RGB32)
        
        pix = widget.grab()
        if pix.isNull():
            return None

        img = pix.toImage().convertToFormat(QImage.Format.Format_RGB32)

        # Ensure the image matches the requested canvas size
        if img.width() != width or img.height() != height:
            img = img.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return img


    # ------------------------------------------------------------------
    # 3D export helpers: stems → RMS envelopes → frames
    # ------------------------------------------------------------------
    def _resolve_visual_stem_path(self, key: str) -> Optional[Path]:
        """
        Translate a visualization routing key into an absolute Path.

        Keys can be:
          - "full_mix"
          - "<model_name>:<stem_name>"
        """
        if self._project is None or not key:
            return None

        if key == "full_mix":
            return self._project.get_audio_path()

        if ":" not in key:
            return None

        model_name, stem_name = key.split(":", 1)
        stems = self._project.get_stems_for_model(model_name)
        rel_path = stems.get(stem_name)
        if not rel_path:
            return None

        return self._project.folder / rel_path

    def _build_visual_input_envelopes(
        self, plugin_id: str, fps: int, main_env: AudioEnvelope
    ) -> Dict[str, np.ndarray]:
        """
        Compute per-input RMS envelopes for a 3D visualization plugin.

        Falls back to the main audio envelope if a stem cannot be loaded.
        """
        envelopes: Dict[str, np.ndarray] = {}

        if self._project is None:
            return envelopes

        visualizations: Dict[str, Dict[str, Any]] = getattr(
            self._project, "visualizations", {}
        ) or {}
        state = visualizations.get(plugin_id, {})
        routing = state.get("routing") or {}

        if not routing:
            envelopes["input_1"] = main_env.rms
            return envelopes

        hop_ms = 1000.0 / float(max(1, fps))

        for input_key, stem_key in routing.items():
            path = self._resolve_visual_stem_path(stem_key)
            if path is None or not path.is_file():
                envelopes[input_key] = main_env.rms
                continue

            try:
                data, sr = sf.read(path, always_2d=True)
            except Exception:
                envelopes[input_key] = main_env.rms
                continue

            if data.dtype != np.float32:
                data = data.astype(np.float32)

            mono = data.mean(axis=1)
            hop_samples = max(1, int(sr * (hop_ms / 1000.0)))
            rms_values: List[float] = []
            for start in range(0, mono.shape[0], hop_samples):
                chunk = mono[start : start + hop_samples]
                if chunk.size == 0:
                    break
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                rms_values.append(min(1.0, rms * 2.0))

            if rms_values:
                envelopes[input_key] = np.array(rms_values, dtype=np.float32)
            else:
                envelopes[input_key] = main_env.rms

        return envelopes

    def _render_visualization_frame(
        self,
        plugin_id: str,
        width: int,
        height: int,
        fps: int,
        time_sec: float,
        envelopes: Dict[str, np.ndarray],
    ) -> Optional[QImage]:
        """
        Render a single 3D visualization frame at the given time.
        """
        if self._project is None:
            return None

        try:
            instance, widget = self._get_or_create_visualization_instance(
                plugin_id, width, height, fps
            )
        except Exception:
            return None

        frame_index = max(0, int(round(time_sec * fps)))

        inputs: Dict[str, Dict[str, float]] = {}
        for input_key, rms_arr in envelopes.items():
            if rms_arr.size == 0:
                amp_val = 0.0
            else:
                idx = min(frame_index, len(rms_arr) - 1)
                amp_val = float(rms_arr[idx])
            inputs[input_key] = {"rms": amp_val}

        features = {
            "time_ms": int(round(time_sec * 1000.0)),
            "inputs": inputs,
        }
        try:
            instance.on_audio_features(features)  # type: ignore[attr-defined]
        except Exception:
            pass

        QApplication.processEvents()
        pix = widget.grab()
        if pix.isNull():
            return None

        img = pix.toImage().convertToFormat(QImage.Format.Format_RGB32)
        
        pix = widget.grab()
        if pix.isNull():
            return None

        img = pix.toImage().convertToFormat(QImage.Format.Format_RGB32)

        # Ensure the image matches the requested canvas size
        if img.width() != width or img.height() != height:
            img = img.scaled(
                width,
                height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        return img

    # ------------------------------------------------------------------
    # Lyrics overlay helpers
    # ------------------------------------------------------------------
    def _lyrics_overlay_enabled(self) -> bool:
        """
        Return True if lyrics overlay is enabled and a plugin id is selected.
        """
        if not hasattr(self, "chk_enable_lyrics_overlay") or not hasattr(
            self, "combo_lyrics_plugin"
        ):
            return False
        if not self.chk_enable_lyrics_overlay.isChecked():
            return False
        plugin_id = self.combo_lyrics_plugin.currentData()
        if not isinstance(plugin_id, str):
            return False
        if not self._lyrics_phrases:
            # No alignment -> nothing meaningful to display
            return False
        return True


    def _get_lyrics_audio_source_id(self, plugin_id: str) -> str:
        """
        Determine which audio source should drive the lyrics amplitude.

        Priority:
          - project.lyrics_visual_routing_by_plugin[plugin_id]["audio_source"]
          - project.lyrics_visual_routing["audio_source"]
          - fallback: "main"
        """
        if self._project is None:
            return "main"

        routing_by_plugin = getattr(
            self._project, "lyrics_visual_routing_by_plugin", {}
        ) or {}
        global_routing = getattr(self._project, "lyrics_visual_routing", {}) or {}

        routing = routing_by_plugin.get(plugin_id) or global_routing
        if isinstance(routing, dict):
            return str(routing.get("audio_source", "main"))
        return "main"

    def _get_or_create_lyrics_instance(
        self,
        plugin_id: str,
        width: int,
        height: int,
    ) -> Optional[BaseLyricsVisualization]:
        """
        Lazily create (or reuse) an off-screen lyrics visualization widget
        for the given plugin id and size.

        The plugin configuration is synced from the project metadata so that
        export uses the same settings as the Lyrics tab.
        """
        if self._project is None:
            return None

        instance = self._lyrics_instances.get(plugin_id)
        if not isinstance(instance, BaseLyricsVisualization):
            # Load configuration from project
            params_by_plugin = getattr(
                self._project, "lyrics_visual_parameters_by_plugin", {}
            ) or {}
            global_params = getattr(
                self._project, "lyrics_visual_parameters", {}
            ) or {}
            config = dict(params_by_plugin.get(plugin_id) or global_params or {})

            instance = self._lyrics_manager.create_instance(
                plugin_id, config=config, parent=self
            )
            if instance is None:
                return None

            # Inject cover pixmap if available (for background_mode == 'cover')
            cover_path = self._project.get_cover_path()
            if cover_path is not None and cover_path.is_file():
                try:
                    pm = QPixmap(str(cover_path))
                    if not pm.isNull():
                        instance.cover_pixmap = pm  # type: ignore[attr-defined]
                except Exception:
                    pass

            instance.setFixedSize(width, height)
            instance.move(-width - 100, -height - 100)
            instance.show()

            self._lyrics_instances[plugin_id] = instance
        else:
            # Resync configuration from project on reuse
            try:
                params_by_plugin = getattr(
                    self._project, "lyrics_visual_parameters_by_plugin", {}
                ) or {}
                global_params = getattr(
                    self._project, "lyrics_visual_parameters", {}
                ) or {}
                config = dict(params_by_plugin.get(plugin_id) or global_params or {})
                instance.load_state(config)
            except Exception:
                pass

            instance.setFixedSize(width, height)
            instance.move(-width - 100, -height - 100)

        return instance

    def _apply_lyrics_overlay(
        self,
        base_img: QImage,
        time_sec: float,
        fps: int,
    ) -> QImage:
        """
        Render the selected lyrics plugin at the given time and composite it
        over the provided base image.

        If overlay is disabled or an error occurs, the base image is returned
        unchanged.
        """
        if not self._lyrics_overlay_enabled():
            return base_img

        plugin_id = self.combo_lyrics_plugin.currentData()
        if not isinstance(plugin_id, str):
            return base_img

        if self._project is None:
            return base_img

        # Determine audio source and amplitude at time_sec
        source_id = self._get_lyrics_audio_source_id(plugin_id)
        audio_path = self._resolve_cover_audio_source_path(source_id)
        if audio_path is None:
            amp = 0.0
        else:
            try:
                env = self._ensure_audio_envelope(audio_path, fps)
                amp = self._sample_envelope(env, time_sec)
            except Exception:
                amp = 0.0

        # Build context from alignment
        ctx = self._build_lyrics_context_for_time(time_sec, amp)

        # Instantiate / reuse plugin
        instance = self._get_or_create_lyrics_instance(
            plugin_id,
            base_img.width(),
            base_img.height(),
        )
        if instance is None:
            return base_img

        # --------------------------------------------------------------
        # Key point for correct compositing:
        # We force the lyrics plugin to use the *current visual frame*
        # (2D cover chain or 3D visualization) as its "cover" background.
        #
        # In practice, lyrics plugins that support background_mode=='cover'
        # will draw this pixmap as their background, then render the text
        # on top. This ensures we keep all 2D/3D effects and simply add
        # the lyrics over them.
        # --------------------------------------------------------------
        try:
            pm = QPixmap.fromImage(base_img)
            if not pm.isNull():
                instance.cover_pixmap = pm  # type: ignore[attr-defined]
        except Exception:
            # Never crash export if cover injection fails
            pass

        try:
            instance.update_frame(ctx)
        except Exception:
            return base_img

        QApplication.processEvents()

        try:
            overlay_pixmap = instance.grab()
            overlay_img = overlay_pixmap.toImage().convertToFormat(
                QImage.Format.Format_ARGB32
            )
        except Exception:
            return base_img

        if overlay_img.isNull():
            return base_img

        # Ensure same size
        if (
            overlay_img.width() != base_img.width()
            or overlay_img.height() != base_img.height()
        ):
            overlay_img = overlay_img.scaled(
                base_img.width(),
                base_img.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Composite: base as ARGB32 + overlay on top
        base_argb = base_img.convertToFormat(QImage.Format.Format_ARGB32)
        result = QImage(base_argb.size(), QImage.Format.Format_ARGB32)
        result.fill(0)

        painter = QPainter(result)
        try:
            painter.drawImage(0, 0, base_argb)
            painter.drawImage(0, 0, overlay_img)
        finally:
            painter.end()

        return result

    def _apply_lyrics_overlay_for_export(
        self,
        base_img: QImage,
        time_sec: float,
        fps: int,
        enabled: bool,
        plugin_id: Optional[str],
        env: Optional[AudioEnvelope],
    ) -> QImage:
        """
        Apply lyrics overlay using settings frozen at export start.

        This avoids reading combo boxes / checkboxes during a running export.
        """
        if not enabled or plugin_id is None or env is None:
            return base_img

        if self._project is None:
            return base_img

        # Sample amplitude from precomputed envelope
        amp = self._sample_envelope(env, time_sec)

        # Build context from alignment
        ctx = self._build_lyrics_context_for_time(time_sec, amp)

        # Instantiate / reuse plugin
        instance = self._get_or_create_lyrics_instance(
            plugin_id,
            base_img.width(),
            base_img.height(),
        )
        if instance is None:
            return base_img

        # Inject current visual frame as cover background
        try:
            pm = QPixmap.fromImage(base_img)
            if not pm.isNull():
                instance.cover_pixmap = pm  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            instance.update_frame(ctx)
        except Exception:
            return base_img

        QApplication.processEvents()

        try:
            overlay_pixmap = instance.grab()
            overlay_img = overlay_pixmap.toImage().convertToFormat(
                QImage.Format.Format_ARGB32
            )
        except Exception:
            return base_img

        if overlay_img.isNull():
            return base_img

        if (
            overlay_img.width() != base_img.width()
            or overlay_img.height() != base_img.height()
        ):
            overlay_img = overlay_img.scaled(
                base_img.width(),
                base_img.height(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        base_argb = base_img.convertToFormat(QImage.Format.Format_ARGB32)
        result = QImage(base_argb.size(), QImage.Format.Format_ARGB32)
        result.fill(0)

        painter = QPainter(result)
        try:
            painter.drawImage(0, 0, base_argb)
            painter.drawImage(0, 0, overlay_img)
        finally:
            painter.end()

        return result

    # ------------------------------------------------------------------
    # Base frame (raw cover or neutral), with same cropping logic
    # ------------------------------------------------------------------
    def _make_base_frame(self, width: int, height: int) -> QImage:
        """
        Fallback: cover cropped to target ratio (no effects), or dark
        background if no cover.
        """
        cover_path = self._project.get_cover_path() if self._project else None
        if cover_path and cover_path.is_file():
            pix = QPixmap(str(cover_path))
            if not pix.isNull():
                src_w = pix.width()
                src_h = pix.height()
                crop_rect = self._compute_crop_rect(src_w, src_h, width, height)
                cropped = pix.copy(crop_rect)
                img = cropped.toImage().convertToFormat(QImage.Format.Format_RGB32)
                if img.width() != width or img.height() != height:
                    img = img.scaled(
                        width,
                        height,
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                return img

        img = QImage(width, height, QImage.Format.Format_RGB32)
        img.fill(Qt.GlobalColor.black)
        return img

    def _on_full_song_clicked(self) -> None:
        """
        Set the duration spin box so that the video covers the full song
        from the current start time.

        We reuse the cached audio envelope to avoid re-doing heavy analysis
        when possible.
        """
        if self._project is None:
            return

        try:
            audio_path = self._project.get_audio_path()
        except Exception:
            audio_path = None

        if audio_path is None or not audio_path.is_file():
            QMessageBox.warning(
                self,
                "Audio not found",
                "Could not resolve the main audio file for this project.",
            )
            return

        # Use a modest FPS for the envelope – we only need duration.
        try:
            env = self._ensure_audio_envelope(audio_path, fps=10)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Audio analysis error",
                f"Could not analyse audio to determine duration:\n{exc}",
            )
            return

        duration = float(getattr(env, "duration", 0.0))
        if duration <= 0.0:
            # Fallback if duration is not stored explicitly
            try:
                rms = getattr(env, "rms", None)
                fps_env = float(getattr(env, "fps", 0.0))
                if rms is not None and fps_env > 0.0:
                    duration = float(len(rms)) / fps_env
            except Exception:
                duration = 0.0

        if duration <= 0.0:
            QMessageBox.warning(
                self,
                "Invalid audio",
                "Could not determine a valid duration for the audio file.",
            )
            return

        start = max(0.0, float(self.spin_start_sec.value()))
        remaining = max(1.0, duration - start)

        max_spin = float(self.spin_duration_sec.maximum())
        new_duration = int(min(max_spin, remaining))
        if new_duration <= 0:
            QMessageBox.warning(
                self,
                "Invalid range",
                "The current start time is beyond the end of the audio.",
            )
            return

        self.spin_duration_sec.setValue(new_duration)

    def _on_stop_export_clicked(self) -> None:
        """
        Stop the current export and discard all frames generated so far.
        """
        if not self._export_cancelled:
            self._export_cancelled = True
            self._export_keep_partial = False
            self.progress_export.setText("Stopping export (discarding frames)...")

    def _on_stop_and_encode_clicked(self) -> None:
        """
        Stop frame generation but still encode a video from the frames
        that have already been rendered.
        """
        if not self._export_cancelled:
            self._export_cancelled = True
            self._export_keep_partial = True
            self.progress_export.setText(
                "Stopping frame rendering, will encode partial video..."
            )

    # ------------------------------------------------------------------
    # Export: full 2D video (existing behavior)
    # ------------------------------------------------------------------
    def _on_export_clicked(self) -> None:
        """
        Export the video using the selected visual and settings.

        This method covers the 2D cover pipeline; 3D visualizations are
        dispatched to _export_3d_video() at the beginning.
        """
        if self._project is None:
            QMessageBox.warning(
                self,
                "No project",
                "Please select a project before exporting.",
            )
            return

        self._export_cancelled = False
        self._export_keep_partial = False
        self.btn_export.setEnabled(False)
        self.btn_stop_export.setEnabled(True)
        self.btn_stop_and_encode.setEnabled(True)
        self.progress_export.setText("Preparing export...")
        QApplication.processEvents()


        try:
            # Determine whether we export a 2D cover video or a 3D visualization.
            mode = self.combo_visual_2d.currentData()
            if isinstance(mode, str) and mode.startswith("3d:"):
                plugin_id = mode.split(":", 1)[1]
                self._export_3d_video(plugin_id)
                return

            chain = getattr(self._project, "cover_visual_chain", None) or []
            effects_cfg = getattr(self._project, "cover_visual_effects", {}) or {}
            if not chain:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.warning(
                    self,
                    "No 2D chain",
                    "This project has no cover_visual_chain defined.",
                )
                return

            if not effects_cfg:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.warning(
                    self,
                    "No 2D effects",
                    "This project has no cover_visual_effects configuration.",
                )
                return

            audio_path = self._project.get_audio_path()
            if audio_path is None or not audio_path.is_file():
                QMessageBox.warning(
                    self,
                    "Audio not found",
                    "Could not resolve the main audio file for this project.",
                )
                return

            fps = max(1, self.spin_fps.value())

            # Canvas = résolution interne de rendu (2D/3D)
            canvas_w, canvas_h = self._get_canvas_size()

            # Output = résolution finale encodée
            width, height = self._get_output_size()

            # Freeze framing rectangle and lyrics settings for this export
            frame_rect = self._compute_output_frame_rect(canvas_w, canvas_h, width, height)

            lyrics_enabled = self._lyrics_overlay_enabled()
            lyrics_plugin_id: Optional[str] = None
            lyrics_env: Optional[AudioEnvelope] = None

            if lyrics_enabled:
                plugin_data = self.combo_lyrics_plugin.currentData()
                if isinstance(plugin_data, str):
                    lyrics_plugin_id = plugin_data
                    source_id = self._get_lyrics_audio_source_id(lyrics_plugin_id)
                    audio_path_lyrics = self._resolve_cover_audio_source_path(source_id)
                    if audio_path_lyrics is not None and audio_path_lyrics.is_file():
                        try:
                            lyrics_env = self._ensure_audio_envelope(
                                audio_path_lyrics, fps
                            )
                        except Exception:
                            lyrics_env = None
                    else:
                        lyrics_env = None
                else:
                    lyrics_enabled = False

            start_sec = max(0, int(self.spin_start_sec.value()))
            duration_sec = max(1, int(self.spin_duration_sec.value()))

            self.progress_export.setText("Computing envelopes...")
            QApplication.processEvents()

            main_env = self._ensure_audio_envelope(audio_path, fps)

            if main_env.duration <= 0.0:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.critical(
                    self,
                    "Envelope error",
                    "The main audio envelope is empty; cannot export video.",
                )
                return

            if start_sec >= main_env.duration:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.warning(
                    self,
                    "Invalid start time",
                    f"Start time ({start_sec}s) is beyond audio duration ({main_env.duration:.1f}s).",
                )
                return

            max_duration = max(0.0, main_env.duration - float(start_sec))
            effective_duration = min(float(duration_sec), max_duration)
            if effective_duration <= 0.0:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.warning(
                    self,
                    "Invalid duration",
                    "Requested duration is zero or exceeds the audio length.",
                )
                return

            n_frames = int(effective_duration * fps)
            if n_frames <= 0:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.warning(
                    self,
                    "Too short",
                    "The selected section is shorter than one frame at the chosen FPS.",
                )
                return

            self.progress_export.setText("Building per-effect envelopes...")
            QApplication.processEvents()

            # For each effect instance in the chain, determine which audio
            # source it is routed to (main or a specific stem).
            entry_to_source: Dict[str, str] = {}
            for entry_key in chain:
                cfg_entry = effects_cfg.get(entry_key)
                if not isinstance(cfg_entry, dict):
                    continue

                routing = cfg_entry.get("routing", {}) or {}
                src_id = routing.get("audio_source", "main")
                entry_to_source[entry_key] = src_id

            # Build / cache envelopes per *distinct* source id
            envelopes_by_source: Dict[str, AudioEnvelope] = {}
            envelopes_by_source["main"] = main_env

            for src_id in set(entry_to_source.values()):
                if src_id in envelopes_by_source:
                    continue
                if src_id == "main":
                    continue

                src_path = self._resolve_cover_audio_source_path(src_id)
                if src_path is None or not src_path.is_file():
                    # Fallback to main audio if stem is missing
                    envelopes_by_source[src_id] = main_env
                    continue

                try:
                    env = self._ensure_audio_envelope(src_path, fps)
                except Exception:
                    env = main_env

                envelopes_by_source[src_id] = env

            default_ext = ".mp4"
            if self._project.name:
                default_name = f"{self._project.name}{default_ext}"
            else:
                default_name = f"export{default_ext}"

            default_path = self._project.folder / default_name  # type: ignore[union-attr]

            out_str, _ = QFileDialog.getSaveFileName(
                self,
                "Export 2D video",
                str(default_path),
                "MP4 video (*.mp4);;All files (*.*)",
            )
            if not out_str:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                return

            output_path = Path(out_str)

            frames_dir = self._project.folder / "_olaf_2d_frames"  # type: ignore[union-attr]
            if frames_dir.exists():
                try:
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass
            frames_dir.mkdir(parents=True, exist_ok=True)

            start_time = float(start_sec)
            for idx in range(n_frames):
                if self._export_cancelled:
                    break

                t = start_time + idx / float(fps)

                amp_by_entry: Dict[str, float] = {}
                for entry_key in chain:
                    src_id = entry_to_source.get(entry_key, "main")
                    env_src = envelopes_by_source.get(src_id, main_env)
                    amp_by_entry[entry_key] = self._sample_envelope(env_src, t)

                # 1) Canvas frame (2D cover chain à la résolution du canvas)
                canvas_img = self._render_cover_chain_frame(
                    width=canvas_w,
                    height=canvas_h,
                    t=t,
                    amp=1.0,
                    amp_by_entry=amp_by_entry,
                )
                if canvas_img is None:
                    canvas_img = self._make_base_frame(canvas_w, canvas_h)

                # 2) Cadrage canvas -> frame de sortie (output size)
                out_img = self._extract_output_frame_from_canvas(
                    canvas_img,
                    frame_rect,
                    width,
                    height,
                )

                # 3) Overlay paroles dans la résolution finale (settings figés)
                out_img = self._apply_lyrics_overlay_for_export(
                    out_img,
                    time_sec=t,
                    fps=fps,
                    enabled=lyrics_enabled,
                    plugin_id=lyrics_plugin_id,
                    env=lyrics_env,
                )

                # Live feedback: show the current frame in the preview label
                # (throttled to reduce CPU/GPU cost).
                try:
                    if idx % 10 == 0 or idx == n_frames - 1:
                        pix = QPixmap.fromImage(out_img)
                        target_w = self.preview_label.width()
                        target_h = self.preview_label.height()
                        if target_w > 0 and target_h > 0:
                            pix = pix.scaled(
                                target_w,
                                target_h,
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.FastTransformation,
                            )
                        self.preview_label.setPixmap(pix)
                        self.preview_label.setText("")
                except Exception:
                    # Preview must never break the export loop
                    pass

                frame_path = frames_dir / f"frame_{idx:06d}.png"
                out_img.save(str(frame_path))

                progress = 5 + int(80.0 * (idx + 1) / n_frames)
                self.progress_export.setText(f"Rendering cover frames... ({progress}%)")
                QApplication.processEvents()

            # How many frames did we actually generate on disk?
            generated_frames = len(list(frames_dir.glob("frame_*.png")))

            if self._export_cancelled and not self._export_keep_partial:
                # Hard cancel -> discard frames and exit
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                self.progress_export.setText("Export cancelled.")
                try:
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass
                return

            if self._export_cancelled and self._export_keep_partial:
                # Partial export -> adjust duration to match rendered frames
                if generated_frames <= 0:
                    self.btn_export.setEnabled(True)
                    self.btn_stop_export.setEnabled(False)
                    self.btn_stop_and_encode.setEnabled(False)

                    self.progress_export.setText("No frames rendered, export cancelled.")
                    try:
                        shutil.rmtree(frames_dir)
                    except Exception:
                        pass
                    return

                effective_duration = generated_frames / float(fps)
                n_frames = generated_frames

            self.progress_export.setText("Encoding video via ffmpeg...")
            QApplication.processEvents()


            try:
                import subprocess

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(frames_dir / "frame_%06d.png"),
                    "-ss",
                    f"{start_time:.3f}",
                    "-t",
                    f"{effective_duration:.3f}",
                    "-i",
                    str(audio_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "320k",
                    "-shortest",
                    str(output_path),
                ]

                result = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        result.stderr.decode(errors="ignore") or "ffmpeg failed"
                    )
            except FileNotFoundError:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.critical(
                    self,
                    "ffmpeg not found",
                    "ffmpeg could not be found on this system.\n"
                    "Please install ffmpeg and ensure it is available in PATH.",
                )
                return
            except Exception as exc:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                QMessageBox.critical(
                    self,
                    "Export error",
                    f"ffmpeg reported an error:\n{exc}",
                )
                return

            self.progress_export.setText("Cleaning up...")
            QApplication.processEvents()

            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass

            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.information(
                self,
                "Export complete",
                f"2D video successfully exported to:\n{output_path}",
            )

        except Exception as exc:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            tb = traceback.format_exc()
            print("ExportTab fatal error in _on_export_clicked:\n", tb, file=sys.stderr)
            QMessageBox.critical(
                self,
                "Unexpected error during export",
                f"{exc}\n\nFull traceback:\n{tb}",
            )

    # ------------------------------------------------------------------
    # Export helper pour les visualisations 3D
    # ------------------------------------------------------------------
    def _export_3d_video(self, plugin_id: str) -> None:
        """
        Export a video driven by a 3D visualization plugin.

        Frames are rendered off-screen from the selected plugin using
        RMS envelopes according to its saved stem routing. Audio is the
        main project mix, trimmed to the selected section.
        """
        if self._project is None:
            QMessageBox.warning(
                self,
                "No project",
                "Please select a project before exporting.",
            )
            return

        visualizations: Dict[str, Dict[str, Any]] = getattr(
            self._project, "visualizations", {}
        ) or {}
        if plugin_id not in visualizations:
            QMessageBox.warning(
                self,
                "No 3D visualization",
                "The selected 3D visualization is not stored in this project.",
            )
            return

        audio_path = self._project.get_audio_path()
        if audio_path is None or not audio_path.is_file():
            QMessageBox.warning(
                self,
                "Audio not found",
                "Could not resolve the main audio file for this project.",
            )
            return

        fps = max(1, self.spin_fps.value())

        # Canvas = résolution interne de rendu 3D
        canvas_w, canvas_h = self._get_canvas_size()

        # Output = résolution finale encodée
        width, height = self._get_output_size()

        # Freeze framing rectangle and lyrics settings for this export
        frame_rect = self._compute_output_frame_rect(canvas_w, canvas_h, width, height)

        lyrics_enabled = self._lyrics_overlay_enabled()
        lyrics_plugin_id: Optional[str] = None
        lyrics_env: Optional[AudioEnvelope] = None

        if lyrics_enabled:
            plugin_data = self.combo_lyrics_plugin.currentData()
            if isinstance(plugin_data, str):
                lyrics_plugin_id = plugin_data
                source_id = self._get_lyrics_audio_source_id(lyrics_plugin_id)
                audio_path_lyrics = self._resolve_cover_audio_source_path(source_id)
                if audio_path_lyrics is not None and audio_path_lyrics.is_file():
                    try:
                        lyrics_env = self._ensure_audio_envelope(audio_path_lyrics, fps)
                    except Exception:
                        lyrics_env = None
                else:
                    lyrics_env = None
            else:
                lyrics_enabled = False

        start_sec = max(0, int(self.spin_start_sec.value()))
        duration_sec = max(1, int(self.spin_duration_sec.value()))

        self._export_cancelled = False
        self._export_keep_partial = False
        self.btn_export.setEnabled(False)
        self.btn_stop_export.setEnabled(True)
        self.btn_stop_and_encode.setEnabled(True)
        self.progress_export.setText("Computing 3D envelopes...")
        QApplication.processEvents()


        try:
            main_env = self._ensure_audio_envelope(audio_path, fps)
        except Exception as exc:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.critical(
                self,
                "Envelope error",
                f"Could not compute RMS envelope for main audio:\n{exc}",
            )
            return

        if main_env.duration <= 0.0:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.critical(
                self,
                "Envelope error",
                "The main audio envelope is empty; cannot export video.",
            )
            return

        if start_sec >= main_env.duration:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.warning(
                self,
                "Invalid start time",
                f"Start time ({start_sec}s) is beyond audio duration ({main_env.duration:.1f}s).",
            )
            return

        max_duration = max(0.0, main_env.duration - float(start_sec))
        effective_duration = min(float(duration_sec), max_duration)
        if effective_duration <= 0.0:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.warning(
                self,
                "Invalid duration",
                "Requested duration is zero or exceeds the audio length.",
            )
            return

        n_frames = int(effective_duration * fps)
        if n_frames <= 0:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.warning(
                self,
                "Too short",
                "The selected section is shorter than one frame at the chosen FPS.",
            )
            return

        envelopes = self._build_visual_input_envelopes(plugin_id, fps, main_env)

        default_ext = ".mp4"
        if self._project.name:
            default_name = f"{self._project.name}_3D_visual{default_ext}"
        else:
            default_name = f"3D_visual{default_ext}"
        default_path = self._project.folder / default_name  # type: ignore[union-attr]

        out_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D video",
            str(default_path),
            "MP4 video (*.mp4);;All files (*.*)",
        )
        if not out_str:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            return

        output_path = Path(out_str)

        frames_dir = self._project.folder / "_olaf_3d_frames"  # type: ignore[union-attr]
        if frames_dir.exists():
            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass
        frames_dir.mkdir(parents=True, exist_ok=True)

        start_time = float(start_sec)
        for idx in range(n_frames):
            if self._export_cancelled:
                break

            t = start_time + idx / float(fps)

            # 1) Canvas frame (3D rendu à la résolution du canvas)
            canvas_img = self._render_visualization_frame(
                plugin_id,
                canvas_w,
                canvas_h,
                fps,
                time_sec=t,
                envelopes=envelopes,
            )
            if canvas_img is None:
                canvas_img = self._make_base_frame(canvas_w, canvas_h)

            # 2) Cadrage canvas -> frame de sortie
            out_img = self._extract_output_frame_from_canvas(
                canvas_img,
                frame_rect,
                width,
                height,
            )

            # 3) Overlay paroles dans la résolution finale (settings figés)
            out_img = self._apply_lyrics_overlay_for_export(
                out_img,
                time_sec=t,
                fps=fps,
                enabled=lyrics_enabled,
                plugin_id=lyrics_plugin_id,
                env=lyrics_env,
            )

            # Live feedback: show the current 3D frame in the preview label
            # (throttled to reduce preview overhead).
            try:
                if idx % 10 == 0 or idx == n_frames - 1:
                    pix = QPixmap.fromImage(out_img)
                    target_w = self.preview_label.width()
                    target_h = self.preview_label.height()
                    if target_w > 0 and target_h > 0:
                        pix = pix.scaled(
                            target_w,
                            target_h,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation,
                        )
                    self.preview_label.setPixmap(pix)
                    self.preview_label.setText("")
            except Exception:
                # Do not interrupt export if preview fails
                pass

            frame_path = frames_dir / f"frame_{idx:06d}.png"
            out_img.save(str(frame_path))

            progress = 5 + int(80.0 * (idx + 1) / n_frames)
            self.progress_export.setText(f"Rendering 3D frames... ({progress}%)")
            QApplication.processEvents()


        # How many frames did we actually render?
        generated_frames = len(list(frames_dir.glob("frame_*.png")))

        if self._export_cancelled and not self._export_keep_partial:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            self.progress_export.setText("3D export cancelled.")
            try:
                shutil.rmtree(frames_dir)
            except Exception:
                pass
            return

        if self._export_cancelled and self._export_keep_partial:
            if generated_frames <= 0:
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setEnabled(False)
                self.btn_stop_and_encode.setEnabled(False)

                self.progress_export.setText("No frames rendered, 3D export cancelled.")
                try:
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass
                return

            effective_duration = generated_frames / float(fps)
            n_frames = generated_frames

        self.progress_export.setText("Encoding 3D video via ffmpeg...")
        QApplication.processEvents()


        try:
            import subprocess

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%06d.png"),
                "-ss",
                f"{start_time:.3f}",
                "-t",
                f"{effective_duration:.3f}",
                "-i",
                str(audio_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "320k",
                "-shortest",
                str(output_path),
            ]

            result = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                message = result.stderr.decode(errors="ignore") or "ffmpeg failed"
                raise RuntimeError(message)
        except FileNotFoundError:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)

            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "ffmpeg could not be found on this system.\n"
                "Please install ffmpeg and ensure it is available in PATH.",
            )
            return
        except Exception as exc:
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setEnabled(False)
            self.btn_stop_and_encode.setEnabled(False)
            QMessageBox.critical(
                self,
                "Export error",
                f"ffmpeg reported an error:\n{exc}",
            )
            return

        self.progress_export.setText("Cleaning up 3D frames...")
        QApplication.processEvents()

        try:
            shutil.rmtree(frames_dir)
        except Exception:
            pass

        self.btn_export.setEnabled(True)
        self.btn_stop_export.setEnabled(False)
        self.btn_stop_and_encode.setEnabled(False)

        QMessageBox.information(
            self,
            "Export complete",
            f"3D video successfully exported to:\n{output_path}",
        )
