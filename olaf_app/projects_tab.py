from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QSettings
from PyQt6.QtGui import QPixmap, QPainter, QBrush, QColor
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QFileDialog,
    QSlider,
    QMessageBox,
    QInputDialog,
    QCheckBox,
    QProgressDialog,
    QApplication,
    QSizePolicy,
)

from .project_manager import Project, list_projects, create_project, delete_project

# ----------------------------------------------------------------------
# Optional auto_clean import
# ----------------------------------------------------------------------

try:
    from . import auto_clean  # type: ignore
    AUTO_CLEAN_AVAILABLE = True
except Exception:
    auto_clean = None  # type: ignore
    AUTO_CLEAN_AVAILABLE = False


class ProjectsTab(QWidget):
    """
    Projects tab:
    - List of projects
    - Basic metadata
    - Cover preview
    - Full-mix playback controls

    Emits:
      projectSelected(Project | None)
      audioStarted(str)  # path of audio when started
    """

    projectSelected = pyqtSignal(object)  # Project or None
    audioStarted = pyqtSignal(str)

    def __init__(self, player: QMediaPlayer, parent=None) -> None:
        super().__init__(parent)
        self.player = player

        self.project_list: QListWidget
        self.lbl_name: QLabel
        self.lbl_id: QLabel
        self.lbl_created: QLabel
        self.lbl_updated: QLabel
        self.lbl_audio: QLabel
        self.lbl_cover: QLabel
        self.cover_label: QLabel

        self.btn_choose_audio: QPushButton
        self.btn_choose_cover: QPushButton
        self.chk_auto_clean: QCheckBox
        self.lbl_auto_clean_summary: QLabel

        self.position_slider: QSlider
        self.lbl_track_time: QLabel

        self._projects_by_name: Dict[str, Project] = {}
        self._current_project: Optional[Project] = None

        # Palette used when there is no background cover
        self._list_default_palette = None

        self._build_ui()

        # After widgets exist
        self._list_default_palette = self.project_list.viewport().palette()

        # Connect shared player callbacks
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

    # ------------------------------------------------------------------ #
    # UI                                                                 #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left side: project list + buttons
        left = QVBoxLayout()
        left.setSpacing(8)

        self.project_list = QListWidget(self)
        self.project_list.itemSelectionChanged.connect(
            self._on_project_selected_in_list
        )

        # Make the list visually clean and let the background cover show through.
        self.project_list.setStyleSheet(
            """
            QListWidget {
                border: none;
                background: transparent;
            }
            QListWidget::item {
                padding: 10px 14px;
            }
            QListWidget::item:selected {
                background: rgba(60, 160, 255, 220);
            }
            """
        )

        left.addWidget(self.project_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_new = QPushButton("New project…", self)
        btn_delete = QPushButton("Delete", self)
        btn_row.addWidget(btn_new)
        btn_row.addWidget(btn_delete)
        left.addLayout(btn_row)

        # Slightly increase the width of the left column (projects list)
        main_layout.addLayout(left, stretch=2)

        # Right side: details, cover, playback
        right = QVBoxLayout()
        right.setSpacing(8)

        self.lbl_name = QLabel("<b>Name:</b> -", self)
        self.lbl_id = QLabel("<b>ID:</b> -", self)
        self.lbl_created = QLabel("<b>Created at:</b> -", self)
        self.lbl_updated = QLabel("<b>Updated at:</b> -", self)
        self.lbl_audio = QLabel("<b>Audio file:</b> (none)", self)
        self.lbl_cover = QLabel("<b>Cover:</b> (none)", self)


        right.addWidget(self.lbl_name)
        right.addWidget(self.lbl_id)
        right.addWidget(self.lbl_created)
        right.addWidget(self.lbl_updated)

        # Audio section
        right.addWidget(self.lbl_audio)

        self.btn_choose_audio = QPushButton("Choose audio…", self)
        right.addWidget(self.btn_choose_audio)

        # Auto-clean checkbox + short summary
        self.chk_auto_clean = QCheckBox("Auto clean on import", self)

        auto_clean_tooltip = (
            "Automatic audio clean-up script for \"boxy / underwater / muffled\" vocal or music recordings.\n\n"
            "This version is especially suited for already-mastered material (e.g. AI-generated\n"
            "songs from Suno): it will NOT increase the overall loudness above the original track.\n"
            "It can only keep it similar or slightly lower.\n\n"
            "Processing chain:\n"
            "1. Optional light noise reduction (noisereduce, spectral gating), only if the file\n"
            "   is not already extremely hot in level.\n"
            "2. Optional peak attenuation at input (only if the file is already very hot).\n"
            "3. Pedalboard effects (tonal shaping + very gentle dynamics):\n"
            "   - High-pass filter at 80 Hz (remove rumble / unnecessary low-end)\n"
            "   - Two low-mid cuts (~300 Hz and ~450 Hz) to reduce \"boxiness\" / \"mud\"\n"
            "   - Simple de-esser: narrow cut around 7 kHz to tame harsh \"s\" sounds\n"
            "   - Presence boost around 3.5 kHz (clarity)\n"
            "   - High-shelf boost from 6 kHz (air / openness)\n"
            "   - Gentle compressor (low ratio, high threshold, mainly catches peaks)\n"
            "   - Very soft noise gate / expander (lightly reduces ambience between phrases)\n"
            "   (No limiter here: we avoid adding extra brickwall limiting.)\n"
            "4. Loudness matching:\n"
            "   - Measure original loudness (after noise reduction)\n"
            "   - Measure processed loudness\n"
            "   - If processed is louder, bring it DOWN to be ~3 dB below the original\n"
            "   - If processed is quieter, leave it as-is (NO upward boost)\n"
            "5. Final peak safety: if peaks exceed ~0 dBFS (|sample| > 0.995), they are attenuated;\n"
            "   otherwise they are untouched.\n\n"
            "Requirements: numpy, pedalboard, noisereduce, pyloudnorm.\n"
            "When enabled, the cleaned file is imported into the project as the main audio."
        )

        if AUTO_CLEAN_AVAILABLE:
            self.chk_auto_clean.setToolTip(auto_clean_tooltip)
        else:
            self.chk_auto_clean.setEnabled(False)
            self.chk_auto_clean.setToolTip(
                auto_clean_tooltip
                + "\n\n[Disabled] auto_clean dependencies are missing "
                  "(e.g. 'noisereduce', 'pedalboard', 'pyloudnorm')."
            )

        right.addWidget(self.chk_auto_clean)

        self.lbl_auto_clean_summary = QLabel(
            "Preset processing to reduce \"boxy / muffled\" sound on mastered tracks\n"
            "(Suno, etc.) while keeping overall loudness similar or slightly lower.",
            self,
        )
        self.lbl_auto_clean_summary.setWordWrap(True)
        self.lbl_auto_clean_summary.setStyleSheet("color: #808080; font-size: 10pt;")
        if not AUTO_CLEAN_AVAILABLE:
            self.lbl_auto_clean_summary.setText(
                self.lbl_auto_clean_summary.text()
                + "\n\nAuto clean is currently disabled (missing dependencies)."
            )
        right.addWidget(self.lbl_auto_clean_summary)

        # Cover section
        right.addWidget(self.lbl_cover)

        self.btn_choose_cover = QPushButton("Choose cover…", self)
        right.addWidget(self.btn_choose_cover)

        # Cover preview
        right.addWidget(QLabel("Cover preview:", self))
        self.cover_label = QLabel(self)
        # Smaller preview: roughly half-column width, square-ish to stay readable
        self.cover_label.setMinimumSize(240, 240)
        self.cover_label.setMaximumWidth(320)
        # Do not allow vertical stretching: keep a compact square-ish preview
        self.cover_label.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        # Image aligned to top-left, no visible border
        self.cover_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.cover_label.setStyleSheet(
            "background-color: transparent; border: none;"
        )
        # Keep it anchored to the left of the column
        right.addWidget(
            self.cover_label,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        # Audio playback controls
        right.addWidget(QLabel("Audio playback:", self))
        audio_row = QHBoxLayout()
        self.btn_play = QPushButton("Play", self)
        self.btn_stop = QPushButton("Stop", self)
        audio_row.addWidget(self.btn_play)
        audio_row.addWidget(self.btn_stop)

        self.position_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.position_slider.setRange(0, 0)
        audio_row.addWidget(self.position_slider, stretch=1)

        self.lbl_track_time = QLabel("0.000 s", self)
        audio_row.addWidget(self.lbl_track_time)

        right.addLayout(audio_row)

        # Push all extra vertical space below the controls,
        # so everything stays aligned to the top.
        right.addStretch(1)

        # Right side slightly wider than left
        main_layout.addLayout(right, stretch=3)

        # Connections
        btn_new.clicked.connect(self._on_new_project_clicked)
        btn_delete.clicked.connect(self._on_delete_project_clicked)
        self.btn_choose_audio.clicked.connect(self._on_choose_audio_clicked)
        self.btn_choose_cover.clicked.connect(self._on_choose_cover_clicked)
        self.btn_play.clicked.connect(self._on_play_clicked)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)

    def _style_project_list_item(self, item: QListWidgetItem) -> None:
        """
        Apply a readable style for project names on top of the cover
        background:
        - larger bold font,
        - semi-transparent dark background,
        - light foreground color.
        """
        base_font = self.project_list.font()
        # Slightly larger and bold
        larger_font = base_font
        larger_font.setPointSize(
            max(base_font.pointSize() + 2, int(base_font.pointSize() * 1.25))
        )
        larger_font.setBold(True)
        item.setFont(larger_font)

        # Semi-transparent dark overlay to keep text readable
        item.setBackground(QColor(0, 0, 0, 160))
        item.setForeground(QColor(240, 240, 240))

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def refresh_projects(self) -> None:
        """Reload projects from disk and refresh the list widget."""
        projects = list_projects()

        self._projects_by_name.clear()
        self.project_list.clear()

        for proj in projects:
            item = QListWidgetItem(proj.name)
            # Apply bigger font + semi-transparent background
            self._style_project_list_item(item)
            self.project_list.addItem(item)
            self._projects_by_name[proj.name] = proj

        if projects:
            self.project_list.setCurrentRow(0)
        else:
            self._current_project = None
            self.show_details(None)
            self.update_cover_preview(None)
            self._update_list_background(None)
            self.projectSelected.emit(None)


    # ------------------------------------------------------------------ #
    # Events                                                             #
    # ------------------------------------------------------------------ #

    def _on_new_project_clicked(self) -> None:
        """Create a new project by asking for a name (no file dialog)."""
        name, ok = QInputDialog.getText(
            self,
            "New project",
            "Project name:",
        )
        if not ok:
            return

        name = name.strip()
        if not name:
            name = "Untitled project"

        project = create_project(name)

        self.refresh_projects()

        # Select the new project
        for row in range(self.project_list.count()):
            if self.project_list.item(row).text() == project.name:
                self.project_list.setCurrentRow(row)
                break

    def _on_delete_project_clicked(self) -> None:
        """Delete currently selected project (after confirmation)."""
        items = self.project_list.selectedItems()
        if not items:
            return

        name = items[0].text()
        project = self._projects_by_name.get(name)
        if not project:
            return

        reply = QMessageBox.question(
            self,
            "Delete project",
            f"Delete project '{project.name}' and all its files on disk?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        delete_project(project)
        self.refresh_projects()

    def _on_project_selected_in_list(self) -> None:
        """When the selection in the project list changes."""
        items = self.project_list.selectedItems()
        if not items:
            self._current_project = None
            self.show_details(None)
            self.update_cover_preview(None)
            self._update_list_background(None)
            self.projectSelected.emit(None)
            return

        name = items[0].text()
        project = self._projects_by_name.get(name)
        self._current_project = project

        self.show_details(project)
        self.update_cover_preview(project)
        self._update_list_background(project)
        self._reset_audio_ui()
        self.projectSelected.emit(project)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._current_project is not None:
            self._update_list_background(self._current_project)

    # ------------------------------------------------------------------ #
    # Background cover for project list                                  #
    # ------------------------------------------------------------------ #

    def _update_list_background(self, project: Optional[Project]) -> None:
        """
        Set the selected project's cover as a semi-transparent background
        behind the project list.
        """
        viewport = self.project_list.viewport()

        if project is None:
            viewport.setAutoFillBackground(False)
            if self._list_default_palette is not None:
                viewport.setPalette(self._list_default_palette)
            return

        cover_path = project.get_cover_path() if project else None
        if not cover_path or not cover_path.is_file():
            viewport.setAutoFillBackground(False)
            if self._list_default_palette is not None:
                viewport.setPalette(self._list_default_palette)
            return

        pix = QPixmap(str(cover_path))
        if pix.isNull():
            viewport.setAutoFillBackground(False)
            if self._list_default_palette is not None:
                viewport.setPalette(self._list_default_palette)
            return

        size = viewport.size()
        if size.width() <= 0 or size.height() <= 0:
            return

        scaled = pix.scaled(
            size,
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )

        x = (scaled.width() - size.width()) // 2
        y = (scaled.height() - size.height()) // 2

        bg = QPixmap(size)
        bg.fill(Qt.GlobalColor.transparent)

        painter = QPainter(bg)
        painter.setOpacity(0.35)
        painter.drawPixmap(-x, -y, scaled)
        painter.end()

        palette = viewport.palette()
        palette.setBrush(viewport.backgroundRole(), QBrush(bg))
        viewport.setPalette(palette)
        viewport.setAutoFillBackground(True)

    # ------------------------------------------------------------------ #
    # Details / cover preview                                            #
    # ------------------------------------------------------------------ #

    def _format_datetime(self, iso_str: str) -> str:
        """
        Convert ISO datetime string -> human-friendly string according to
        user preference stored in QSettings ('ui/date_format').
        """
        if not iso_str:
            return "-"

        try:
            dt = datetime.fromisoformat(iso_str)
        except Exception:
            return iso_str

        settings = QSettings("Olaf", "OlafApp")
        fmt = settings.value("ui/date_format", "%Y-%m-%d %H:%M", type=str)
        try:
            return dt.strftime(fmt)
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M")

    def show_details(self, project: Optional[Project]) -> None:
        if project is None:
            self.lbl_name.setText("<b>Name:</b> -")
            self.lbl_id.setText("<b>ID:</b> -")
            self.lbl_created.setText("<b>Created at:</b> -")
            self.lbl_updated.setText("<b>Updated at:</b> -")
            self.lbl_audio.setText("<b>Audio file:</b> (none)")
            self.lbl_cover.setText("<b>Cover:</b> (none)")
            return

        self.lbl_name.setText(f"<b>Name:</b> {project.name}")
        self.lbl_id.setText(f"<b>ID:</b> {project.id}")
        self.lbl_created.setText(
            f"<b>Created at:</b> {self._format_datetime(project.created_at)}"
        )
        self.lbl_updated.setText(
            f"<b>Updated at:</b> {self._format_datetime(project.updated_at)}"
        )

        audio_path = project.get_audio_path()
        if audio_path and audio_path.is_file():
            self.lbl_audio.setText(f"<b>Audio file:</b> {audio_path}")
        else:
            self.lbl_audio.setText("<b>Audio file:</b> (none)")

        cover_path = project.get_cover_path()
        if cover_path and cover_path.is_file():
            self.lbl_cover.setText(f"<b>Cover:</b> {cover_path}")
        else:
            self.lbl_cover.setText("<b>Cover:</b> (none)")


    def update_cover_preview(self, project: Optional[Project]) -> None:
        if project is None:
            self.cover_label.clear()
            self.cover_label.setText("(no cover)")
            return

        cover_path = project.get_cover_path()
        if not cover_path or not cover_path.is_file():
            self.cover_label.clear()
            self.cover_label.setText("(no cover)")
            return

        pix = QPixmap(str(cover_path))
        if pix.isNull():
            self.cover_label.clear()
            self.cover_label.setText("(invalid image)")
            return

        # Fill the preview area and crop if necessary, anchored at top-left.
        scaled = pix.scaled(
            self.cover_label.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cover_label.setPixmap(scaled)

    # ------------------------------------------------------------------ #
    # Audio playback (local controls, shared player)                     #
    # ------------------------------------------------------------------ #

    def _reset_audio_ui(self) -> None:
        self.position_slider.setRange(0, 0)
        self.lbl_track_time.setText("0.000 s")

    def _on_choose_audio_clicked(self) -> None:
        if not self._current_project:
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files (*.wav *.mp3 *.flac *.m4a *.aac);;All files (*.*)",
        )
        if not path_str:
            return

        src = Path(path_str)
        src_to_copy = src

        # Auto clean only if checkbox is enabled & checked and import is available
        if AUTO_CLEAN_AVAILABLE and self.chk_auto_clean.isChecked():
            progress = QProgressDialog(
                "Cleaning audio, please wait…",
                None,
                0,
                0,
                self,
            )
            progress.setWindowTitle("Auto clean in progress")
            progress.setWindowModality(Qt.WindowModality.ApplicationModal)
            progress.setMinimumDuration(0)
            progress.show()
            QApplication.processEvents()

            try:
                cleaned_path = src.with_name(src.stem + "_cleaned" + src.suffix)
                auto_clean.process_file(str(src), str(cleaned_path))  # type: ignore
                src_to_copy = cleaned_path
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Auto clean failed",
                    f"An error occurred while cleaning the audio:\n{e}\n\n"
                    "The original file will be imported instead.",
                )
                src_to_copy = src
            finally:
                progress.close()
                QApplication.processEvents()

        try:
            self._current_project.set_audio_from_path(src_to_copy)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not copy audio:\n{e}")
            return

        self.show_details(self._current_project)

    def _on_choose_cover_clicked(self) -> None:
        if not self._current_project:
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select cover image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All files (*.*)",
        )
        if not path_str:
            return

        src = Path(path_str)

        try:
            self._current_project.set_cover_from_path(src)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not add cover image:\n{e}")
            return

        self.show_details(self._current_project)
        self.update_cover_preview(self._current_project)
        self._update_list_background(self._current_project)

    def _on_play_clicked(self) -> None:
        if not self._current_project:
            return
        audio_path = self._current_project.get_audio_path()
        if not audio_path or not audio_path.is_file():
            QMessageBox.warning(self, "No audio", "This project has no audio file.")
            return

        url = QUrl.fromLocalFile(str(audio_path.resolve()))
        self.player.setSource(url)
        self.player.play()
        self.audioStarted.emit(str(audio_path))

    def _on_stop_clicked(self) -> None:
        self.player.stop()

    def _on_slider_moved(self, value: int) -> None:
        self.player.setPosition(value)

    # ------------------------------------------------------------------ #
    # Shared player callbacks                                            #
    # ------------------------------------------------------------------ #

    def on_position_changed(self, pos_ms: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(pos_ms)
        self.position_slider.blockSignals(False)
        self.lbl_track_time.setText(f"{pos_ms / 1000.0:.3f} s")

    def on_duration_changed(self, dur_ms: int) -> None:
        self.position_slider.setRange(0, dur_ms)

    def on_media_status_changed(self, status) -> None:
        # Placeholder if you want to show status-dependent info
        pass
