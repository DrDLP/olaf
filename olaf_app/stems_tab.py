from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QDesktopServices
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QComboBox,
    QMessageBox,
    QGroupBox,
    QApplication,
    QInputDialog,
)

from .project_manager import Project
from .stem_separator import separate_stems_for_project


class StemsTab(QWidget):
    """
    Stems tab:
    - Shows stems for the current project
    - Runs Demucs
    - Plays stems via shared QMediaPlayer
    - Displays a waveform + playhead for the selected stem
    - Allows click-to-seek on the waveform
    """
    
    audioStarted = pyqtSignal(str)

    def __init__(self, player: QMediaPlayer, parent=None):
        super().__init__(parent)
        self.player = player
        self._project: Optional[Project] = None

        # For waveform / playhead handling
        self._current_stem_path: Optional[Path] = None
        self._stem_duration_ms: int = 0       # from audio file
        self._waveform_base: Optional[QPixmap] = None  # waveform without playhead

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _refresh_model_labels(self):
        """
        Met à jour le texte des items du combo Demucs pour indiquer
        combien de stems existent pour chaque modèle dans le projet courant.
        """
        for idx in range(self.model_combo.count()):
            model_key = self.model_combo.itemData(idx)  # UserRole (clé du modèle)
            base_label = self.model_combo.itemData(idx, Qt.ItemDataRole.UserRole + 1)
            if not base_label:
                # Sécurité : si jamais ce n'était pas encore stocké, on enlève l'ancien suffixe
                text = self.model_combo.itemText(idx)
                base_label = text.split(" — ", 1)[0]
                self.model_combo.setItemData(idx, base_label, Qt.ItemDataRole.UserRole + 1)

            # Cas sans projet : on indique simplement "no stems"
            if not self._project:
                suffix = " — no stems"
            else:
                stems = self._project.get_stems_for_model(model_key) if model_key else {}
                n = len(stems)
                if n == 0:
                    suffix = " — no stems"
                elif n == 1:
                    suffix = " — 1 stem"
                else:
                    suffix = f" — {n} stems"

            self.model_combo.setItemText(idx, f"{base_label}{suffix}")


    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.lbl_stems_project = QLabel("", self)
        self.lbl_stems_project.setVisible(False)
        layout.addWidget(self.lbl_stems_project)

        self.btn_separate_stems = QPushButton("Separate stems now", self)
        self.btn_separate_stems.clicked.connect(self.run_stem_separation)
        layout.addWidget(self.btn_separate_stems, alignment=Qt.AlignmentFlag.AlignLeft)

        self.stems_progress = QProgressBar(self)
        self.stems_progress.setRange(0, 100)
        self.stems_progress.setValue(0)
        layout.addWidget(self.stems_progress)

        self.lbl_stems_status = QLabel("Ready.", self)
        layout.addWidget(self.lbl_stems_status)

        layout.addWidget(QLabel("Demucs model:", self))
        self.model_combo = QComboBox(self)

        # On utilise un helper pour stocker le label "de base" de chaque modèle
        self._add_model_choice("htdemucs (4 stems: vocals, drums, bass, other)", "htdemucs")
        self._add_model_choice("htdemucs_6s (6 stems: + guitar, piano)", "htdemucs_6s")
        self._add_model_choice("htdemucs_ft (fine-tuned, slower but better)", "htdemucs_ft")
        self._add_model_choice("hdemucs_mmi (v3 hybrid)", "hdemucs_mmi")
        self._add_model_choice("mdx (MDX model)", "mdx")
        self._add_model_choice("mdx_extra (MDX extra data)", "mdx_extra")

        # Si tu ne l’as pas déjà :
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        layout.addWidget(self.model_combo)


        # Quality preset
        layout.addWidget(QLabel("Quality:", self))
        self.quality_combo = QComboBox(self)
        self.quality_combo.addItem("Fast (lower quality, faster)", "fast")
        self.quality_combo.addItem("Balanced (default)", "balanced")
        self.quality_combo.addItem("HQ (better separation, slower)", "hq")
        layout.addWidget(self.quality_combo)

        layout.addWidget(QLabel("Stems:", self))

        self.stems_list = QListWidget(self)
        self.stems_list.itemDoubleClicked.connect(self.on_stem_double_clicked)
        self.stems_list.currentItemChanged.connect(self.on_stem_selection_changed)
        layout.addWidget(self.stems_list)

        # Playback controls + file operations for stems
        btn_row = QHBoxLayout()
        self.btn_play_stem = QPushButton("Play selected stem", self)
        self.btn_stop_stem = QPushButton("Stop", self)
        self.btn_open_folder = QPushButton("Open stem folder", self)
        self.btn_rename_stem = QPushButton("Rename stem", self)
        self.btn_delete_stem = QPushButton("Delete stem", self)
        self.btn_delete_all_stems = QPushButton("Delete all (this model)", self)

        self.btn_play_stem.clicked.connect(self.play_selected_stem)
        self.btn_stop_stem.clicked.connect(self.player.stop)
        self.btn_open_folder.clicked.connect(self.open_stem_folder)
        self.btn_rename_stem.clicked.connect(self.rename_selected_stem)
        self.btn_delete_stem.clicked.connect(self.delete_selected_stem)
        self.btn_delete_all_stems.clicked.connect(self.delete_all_stems_for_model)

        btn_row.addWidget(self.btn_play_stem)
        btn_row.addWidget(self.btn_stop_stem)
        btn_row.addWidget(self.btn_open_folder)
        btn_row.addWidget(self.btn_rename_stem)
        btn_row.addWidget(self.btn_delete_stem)
        btn_row.addWidget(self.btn_delete_all_stems)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # Waveform group
        self.waveform_group = QGroupBox("Stem waveform", self)
        wf_layout = QVBoxLayout(self.waveform_group)
        self.waveform_label = QLabel("No waveform", self.waveform_group)
        self.waveform_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.waveform_label.setFixedHeight(180)
        self.waveform_label.setStyleSheet(
            "border: 1px solid #444; background-color: #111; color: #aaa;"
        )
        wf_layout.addWidget(self.waveform_label)
        layout.addWidget(self.waveform_group)

        # Click-to-seek on waveform
        self.waveform_label.mousePressEvent = self.on_waveform_mouse_press

        layout.addStretch(1)

        self.clear_stems_ui()

    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------

    def set_project(self, project: Optional[Project]):
        """Set the current project whose stems will be shown."""
        self._project = project
        if project:
            self.lbl_stems_project.setText(f"Current project: {project.name}")
        else:
            self.lbl_stems_project.setText("Current project: (none)")
        self.update_stems_ui()
        self.clear_waveform()
        self._refresh_model_labels()


    # These are connected to QMediaPlayer in MainWindow
    def on_position_changed(self, position_ms: int):
        """Update playhead position on waveform when a stem is playing."""
        if not self._current_stem_path:
            return
        if not self._is_player_on_current_stem():
            return
        self.update_waveform_cursor(position_ms)

    def on_duration_changed(self, duration_ms: int):
        """
        Called when QMediaPlayer detects a duration for the current source.
        We prefer the duration computed from the audio file.
        """
        if not self._current_stem_path:
            return
        if not self._is_player_on_current_stem():
            return

        if self._stem_duration_ms <= 0 and duration_ms > 0:
            # Fallback if file-based duration is missing
            self._stem_duration_ms = duration_ms

        self.update_waveform_cursor(self.player.position())

    # ------------------------------------------------------------------
    # Model / quality helpers
    # ------------------------------------------------------------------

    def on_model_changed(self, index: int):
        """When user changes Demucs model in the combo, update stems list for that model."""
        self.update_stems_ui()
        self.clear_waveform()

    def current_model_key(self) -> str:
        """Return the Demucs model key used in metadata."""
        model = self.model_combo.currentData()
        return model or "htdemucs"

    def current_quality_key(self) -> str:
        q = getattr(self, "quality_combo", None)
        if q is None:
            return "balanced"
        return q.currentData() or "balanced"

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _add_model_choice(self, base_label: str, model_key: str):
        """
        Ajoute un modèle dans la combo et mémorise son label de base
        (sans le suffixe "— X stems").
        """
        idx = self.model_combo.count()
        self.model_combo.addItem(base_label, model_key)
        # On stocke le label "de base" dans UserRole+1
        self.model_combo.setItemData(idx, base_label, Qt.ItemDataRole.UserRole + 1)

    def update_stems_ui(self):
        self.stems_list.clear()

        if not self._project:
            placeholder = QListWidgetItem("No stems available.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.stems_list.addItem(placeholder)
            self.clear_waveform()
            return

        model = self.current_model_key()
        stems = self._project.get_stems_for_model(model)

        if not stems:
            placeholder = QListWidgetItem(f"No stems for model '{model}'.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.stems_list.addItem(placeholder)
            self.clear_waveform()
            return

        for name, relpath in sorted(stems.items()):
            text = f"{name}: {relpath}"
            item = QListWidgetItem(text)
            # Store both logical stem name and relpath for later ops
            item.setData(Qt.ItemDataRole.UserRole, relpath)
            item.setData(Qt.ItemDataRole.UserRole + 1, name)
            self.stems_list.addItem(item)

        if self.stems_list.count() > 0:
            self.stems_list.setCurrentRow(0)
            
        self._refresh_model_labels()    

    def clear_stems_ui(self):
        self.stems_progress.setValue(0)
        self.lbl_stems_status.setText("Ready.")
        self.stems_list.clear()
        placeholder = QListWidgetItem("No stems available.")
        placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
        self.stems_list.addItem(placeholder)
        self.clear_waveform()

    def on_stem_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Change of current item in the stems list."""
        # If a stem was playing corresponding to previous selection, stop playback
        if previous is not None and self._current_stem_path is not None:
            if self._is_player_on_current_stem():
                self.player.stop()

        self.load_waveform_for_item(current)

    # ------------------------------------------------------------------
    # Stem file / metadata helpers
    # ------------------------------------------------------------------

    def open_stem_folder(self):
        """Open the folder containing the currently selected stem in the OS file explorer."""
        path = self.get_current_item_path()
        if not path:
            QMessageBox.warning(
                self,
                "No stem selected",
                "Please select a stem from the list first.",
            )
            return

        folder = path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder.resolve())))

    def _path_from_item(self, item: Optional[QListWidgetItem]) -> Optional[Path]:
        """Get absolute Path for a stem from a QListWidgetItem."""
        if not self._project or item is None:
            return None
        relpath = item.data(Qt.ItemDataRole.UserRole)
        if not relpath:
            return None
        path = self._project.folder / str(relpath)
        return path if path.is_file() else None

    def _name_from_item(self, item: Optional[QListWidgetItem]) -> Optional[str]:
        """Return logical stem name stored for this item."""
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole + 1)
        if not isinstance(name, str):
            return None
        return name

    def get_current_item_path(self) -> Optional[Path]:
        """Path of the stem corresponding to the current item."""
        current = self.stems_list.currentItem()
        return self._path_from_item(current)

    def get_current_item_name(self) -> Optional[str]:
        current = self.stems_list.currentItem()
        return self._name_from_item(current)

    # ------------------------------------------------------------------
    # Waveform rendering
    # ------------------------------------------------------------------

    def clear_waveform(self):
        self._waveform_base = None
        self._stem_duration_ms = 0
        self._current_stem_path = None
        self.waveform_label.setPixmap(QPixmap())
        self.waveform_label.setText("No waveform")

    def load_waveform_for_item(self, item: Optional[QListWidgetItem]):
        """Load audio from the given item (stem) and render the waveform."""
        path = self._path_from_item(item)
        if not path:
            self.clear_waveform()
            return

        try:
            data, sr = sf.read(str(path), always_2d=False)
        except Exception as e:
            self.clear_waveform()
            self.waveform_label.setText(f"Failed to read audio: {e}")
            return

        # Convert to mono if needed
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.mean(axis=1)

        if not isinstance(data, np.ndarray) or data.size == 0:
            self.clear_waveform()
            self.waveform_label.setText("Empty audio.")
            return

        samples = data.astype(float)

        # Duration in ms from file
        num_samples = samples.size
        if sr > 0:
            self._stem_duration_ms = int(num_samples * 1000 / sr)
        else:
            self._stem_duration_ms = 0

        # Normalize
        max_val = float(np.max(np.abs(samples)))
        if max_val > 0:
            samples = samples / max_val

        # Display size
        size = self.waveform_label.size()
        width = max(size.width(), 300)
        height = max(size.height(), 120)

        if num_samples < 2:
            self.clear_waveform()
            self.waveform_label.setText("Audio too short.")
            return

        # Min/max envelope per column
        bucket_size = int(np.ceil(num_samples / float(width)))
        if bucket_size < 1:
            bucket_size = 1

        mins = np.zeros(width, dtype=float)
        maxs = np.zeros(width, dtype=float)

        for x in range(width):
            start_idx = x * bucket_size
            if start_idx >= num_samples:
                mins[x] = 0.0
                maxs[x] = 0.0
                continue
            end_idx = min(num_samples, start_idx + bucket_size)
            segment = samples[start_idx:end_idx]
            if segment.size == 0:
                mins[x] = 0.0
                maxs[x] = 0.0
            else:
                mins[x] = float(segment.min())
                maxs[x] = float(segment.max())

        pix = QPixmap(width, height)
        pix.fill(QColor(17, 17, 17))

        painter = QPainter(pix)
        pen = QPen(QColor(0, 220, 255))
        pen.setWidth(1)
        painter.setPen(pen)

        mid = height / 2.0
        scale = height * 0.45

        for x in range(width):
            ymin = mid - maxs[x] * scale
            ymax = mid - mins[x] * scale
            painter.drawLine(x, int(ymin), x, int(ymax))

        painter.end()

        self._waveform_base = pix
        self._current_stem_path = path
        self.waveform_label.setText("")
        self.update_waveform_cursor(0)

    def update_waveform_cursor(self, pos_ms: int):
        """Draw a vertical playhead over the base waveform."""
        if self._waveform_base is None:
            return

        pix = self._waveform_base.copy()

        if self._stem_duration_ms > 0:
            ratio = max(0.0, min(1.0, float(pos_ms) / float(self._stem_duration_ms)))
            x = int(ratio * (pix.width() - 1))

            painter = QPainter(pix)
            pen = QPen(QColor(255, 80, 80))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawLine(x, 0, x, pix.height())
            painter.end()

        self.waveform_label.setPixmap(pix)

    # ------------------------------------------------------------------
    # Click-to-seek in waveform
    # ------------------------------------------------------------------

    def on_waveform_mouse_press(self, event):
        """
        Seek in the current stem by clicking on the waveform.
        If the player is not on this stem, we start playing this stem at the clicked position.
        """
        if self._stem_duration_ms <= 0:
            return
        if not self._project:
            return

        x = event.position().x() if hasattr(event, "position") else event.x()
        width = max(1, self.waveform_label.width())
        ratio = max(0.0, min(1.0, float(x) / float(width)))
        target_ms = int(ratio * self._stem_duration_ms)

        current_item = self.stems_list.currentItem()
        path = self._path_from_item(current_item)
        if not path:
            return

        url = QUrl.fromLocalFile(str(path.resolve()))

        # If player is already on this stem, just seek
        try:
            current_file = self.player.source().toLocalFile()
        except AttributeError:
            current_file = str(self.player.source())

        if current_file and Path(current_file) == path:
            self.player.setPosition(target_ms)
        else:
            # Switch to this stem and start playing from clicked position
            self._current_stem_path = path
            self.player.setSource(url)
            self.player.setPosition(target_ms)
            self.player.play()

            stem_name = self.get_current_item_name() or "(unnamed stem)"
            model = self.current_model_key()
            project_name = self._project.name if self._project else "(no project)"
            self.audioStarted.emit(f"Stem: {project_name} — {model} / {stem_name}")

        self.update_waveform_cursor(target_ms)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_player_on_current_stem(self) -> bool:
        """Check whether QMediaPlayer's current source is the current stem file."""
        if not self._current_stem_path:
            return False

        source = self.player.source()
        try:
            current_file = source.toLocalFile()
        except AttributeError:
            current_file = str(source)

        if not current_file:
            return False

        try:
            return Path(current_file).resolve() == self._current_stem_path.resolve()
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Rename / delete operations
    # ------------------------------------------------------------------

    def rename_selected_stem(self):
        """Rename the logical name (and file) of the selected stem."""
        if not self._project:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        item = self.stems_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "No stem selected", "Please select a stem first.")
            return

        model = self.current_model_key()
        stems = self._project.get_stems_for_model(model)
        if not stems:
            QMessageBox.warning(self, "No stems", "There are no stems for this model.")
            return

        old_name = self._name_from_item(item)
        if not old_name or old_name not in stems:
            QMessageBox.warning(self, "Invalid stem", "Could not determine stem name.")
            return

        relpath = stems[old_name]
        old_path = self._project.folder / relpath
        if not old_path.exists():
            QMessageBox.warning(self, "Missing file", "Stem file does not exist on disk.")
            return

        new_name, ok = QInputDialog.getText(
            self,
            "Rename stem",
            "New stem name:",
            text=old_name,
        )
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid name", "Stem name cannot be empty.")
            return
        if new_name == old_name:
            return
        if new_name in stems:
            QMessageBox.warning(self, "Already exists", "Another stem has this name.")
            return

        # Compute new file path: same folder, new basename, same extension
        suffix = old_path.suffix
        new_path = old_path.with_name(new_name + suffix)

        try:
            old_path.rename(new_path)
        except Exception as e:
            QMessageBox.critical(self, "Rename error", f"Could not rename stem file: {e}")
            return

        # Update metadata
        new_rel = str(new_path.relative_to(self._project.folder))
        del stems[old_name]
        stems[new_name] = new_rel
        self._project.set_stems_for_model(model, stems)

        # Refresh UI
        self.update_stems_ui()

    def delete_selected_stem(self):
        """Delete the selected stem (file + metadata) for the current model."""
        if not self._project:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        item = self.stems_list.currentItem()
        if item is None:
            QMessageBox.warning(self, "No stem selected", "Please select a stem first.")
            return

        model = self.current_model_key()
        stems = self._project.get_stems_for_model(model)
        if not stems:
            QMessageBox.warning(self, "No stems", "There are no stems for this model.")
            return

        name = self._name_from_item(item)
        if not name or name not in stems:
            QMessageBox.warning(self, "Invalid stem", "Could not determine stem name.")
            return

        reply = QMessageBox.question(
            self,
            "Delete stem",
            f"Delete stem '{name}' for model '{model}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        relpath = stems[name]
        path = self._project.folder / relpath

        try:
            if path.exists():
                path.unlink()
        except Exception as e:
            QMessageBox.warning(self, "File error", f"Could not delete stem file: {e}")

        # Update metadata
        del stems[name]
        self._project.set_stems_for_model(model, stems)
        self.update_stems_ui()

    def delete_all_stems_for_model(self):
        """Delete all stems (files + metadata) for the current model."""
        if not self._project:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        model = self.current_model_key()
        stems = self._project.get_stems_for_model(model)
        if not stems:
            QMessageBox.information(self, "No stems", f"No stems found for model '{model}'.")
            return

        reply = QMessageBox.question(
            self,
            "Delete all stems",
            f"Delete ALL stems for model '{model}'?\n"
            "This will remove corresponding files on disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete files
        for relpath in list(stems.values()):
            path = self._project.folder / relpath
            try:
                if path.exists():
                    path.unlink()
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "File error",
                    f"Could not delete stem file '{path.name}': {e}",
                )

        # Clear metadata for this model
        self._project.set_stems_for_model(model, {})
        self.update_stems_ui()

    # ------------------------------------------------------------------
    # Stem playback
    # ------------------------------------------------------------------

    def play_selected_stem(self):
        if not self._project:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        path = self.get_current_item_path()
        if not path:
            QMessageBox.warning(
                self,
                "No stem selected",
                "Please select a stem from the list (after separation).",
            )
            return

        # We assume waveform already corresponds to this stem
        self._current_stem_path = path
        self.update_waveform_cursor(0)

        url = QUrl.fromLocalFile(str(path.resolve()))
        self.player.setSource(url)
        self.player.setPosition(0)
        self.player.play()
        
        stem_name = self.get_current_item_name() or "(unnamed stem)"
        model = self.current_model_key()
        project_name = self._project.name if self._project else "(no project)"
        self.audioStarted.emit(f"Stem: {project_name} — {model} / {stem_name}")        

    def on_stem_double_clicked(self, item: QListWidgetItem):
        # The double-clicked item becomes current; Qt handles that.
        item.setSelected(True)
        self.play_selected_stem()

    # ------------------------------------------------------------------
    # Separation
    # ------------------------------------------------------------------

    def run_stem_separation(self):
        if not self._project:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        audio_path = self._project.get_audio_path()
        if not audio_path or not audio_path.is_file():
            QMessageBox.warning(
                self,
                "No audio file",
                "No audio file is associated with this project or the file is missing.",
            )
            return

        self.btn_separate_stems.setEnabled(False)
        self.btn_separate_stems.setText("Separating…")
        self.stems_progress.setValue(0)
        self.lbl_stems_status.setText("Starting stem separation...")
        self.stems_list.clear()
        QApplication.processEvents()

        model = self.current_model_key()
        quality = self.current_quality_key()

        def progress_cb(percent: float, message: str):
            self.stems_progress.setValue(int(percent))
            self.lbl_stems_status.setText(message[:150])
            QApplication.processEvents()

        try:
            separate_stems_for_project(
                self._project,
                model=model,
                quality=quality,
                progress_cb=progress_cb,
            )
        except Exception as e:
            QMessageBox.critical(self, "Stem separation error", f"Could not separate stems: {e}")
        finally:
            self.btn_separate_stems.setEnabled(True)
            self.btn_separate_stems.setText("Separate stems now")

        self.update_stems_ui()
