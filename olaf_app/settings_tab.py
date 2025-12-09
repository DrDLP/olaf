from __future__ import annotations

from PyQt6.QtCore import QSettings, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGroupBox,
    QFormLayout,
    QSpinBox,
    QComboBox,
    QSlider,
    QHBoxLayout,
)


class SettingsTab(QWidget):
    """Application-wide preferences."""

    # Emitted when the selected UI theme changes ("dark", "light", "neon", ...).
    themeChanged = pyqtSignal(str)
    # Emitted when the user adjusts the global UI scale (percentage).
    scaleChanged = pyqtSignal(int)
    # Global audio volume (0–100 %), used by the shared QAudioOutput.
    volumeChanged = pyqtSignal(int)

    """
    Application-wide preferences.

    For now this tab exposes:
      - Default preview settings for visualization plugins
      - Date/time display format used in the Projects tab
      - Global UI theme and UI scale
      - Global audio playback volume
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # QSettings: persisted across runs
        self._settings = QSettings("Olaf", "OlafApp")

        layout = QVBoxLayout(self)

        header = QLabel("Application settings", self)
        layout.addWidget(header)

        # ---- User interface settings ----------------------------------
        ui_group = QGroupBox("User interface", self)
        ui_form = QFormLayout(ui_group)

        # Theme selection
        self.combo_theme = QComboBox(ui_group)
        self.combo_theme.addItem("Dark (neutral)", "dark")
        self.combo_theme.addItem("Light", "light")
        self.combo_theme.addItem("Neon blue studio", "neon")

        stored_theme = self._settings.value("ui/theme", "neon", type=str)
        theme_index = 0
        for i in range(self.combo_theme.count()):
            if self.combo_theme.itemData(i) == stored_theme:
                theme_index = i
                break
        self.combo_theme.setCurrentIndex(theme_index)
        ui_form.addRow("Color theme:", self.combo_theme)

        # Global UI scale
        self.combo_scale = QComboBox(ui_group)
        self._scale_values = [75, 90, 100, 110, 125]
        for pct in self._scale_values:
            self.combo_scale.addItem(f"{pct} %", pct)

        stored_scale = int(self._settings.value("ui/scale_percent", 100))
        scale_index = 0
        for i, pct in enumerate(self._scale_values):
            if pct == stored_scale:
                scale_index = i
                break
        self.combo_scale.setCurrentIndex(scale_index)
        ui_form.addRow("UI scale:", self.combo_scale)

        # Date/time display format
        self.combo_date_format = QComboBox(ui_group)
        # (strftime pattern, example label)
        self._date_formats = [
            ("%Y-%m-%d %H:%M", "2025-11-30 14:45"),
            ("%d.%m.%Y %H:%M", "30.11.2025 14:45"),
            ("%d/%m/%Y %H:%M", "30/11/2025 14:45"),
        ]
        for fmt, example in self._date_formats:
            self.combo_date_format.addItem(example, fmt)

        stored_fmt = self._settings.value("ui/date_format", "%Y-%m-%d %H:%M", type=str)
        index_to_set = 0
        for i, (fmt, _) in enumerate(self._date_formats):
            if fmt == stored_fmt:
                index_to_set = i
                break
        self.combo_date_format.setCurrentIndex(index_to_set)

        ui_form.addRow("Date & time format:", self.combo_date_format)

        ui_info = QLabel(
            "Controls how dates such as 'Created at' or 'Updated at'\n"
            "are displayed in the Projects tab.",
            ui_group,
        )
        ui_info.setWordWrap(True)
        ui_form.addRow(ui_info)

        ui_group.setLayout(ui_form)
        layout.addWidget(ui_group)

        # ---- Audio / playback -----------------------------------------
        audio_group = QGroupBox("Audio / playback", self)
        audio_form = QFormLayout(audio_group)

        # Global audio volume slider (0–100 %), persisted in QSettings
        self.volume_slider = QSlider(Qt.Orientation.Horizontal, audio_group)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setSingleStep(5)

        stored_volume = self._settings.value("audio/volume_percent", 100, type=int)
        try:
            vol = int(stored_volume)
        except Exception:
            vol = 100
        vol = max(0, min(100, vol))
        self.volume_slider.setValue(vol)

        self.lbl_volume_value = QLabel(f"{vol} %", audio_group)

        # Put slider + numeric label on the same row
        row_widget = QWidget(audio_group)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.volume_slider)
        row_layout.addWidget(self.lbl_volume_value)

        audio_form.addRow("Global volume:", row_widget)
        audio_group.setLayout(audio_form)
        layout.addWidget(audio_group)

        # ---- Visualization preview defaults ---------------------------
        preview_group = QGroupBox("Visualization preview defaults", self)
        form = QFormLayout(preview_group)

        self.spin_preview_width = QSpinBox(preview_group)
        self.spin_preview_width.setRange(160, 1920)

        self.spin_preview_height = QSpinBox(preview_group)
        self.spin_preview_height.setRange(90, 1080)

        self.spin_preview_fps = QSpinBox(preview_group)
        self.spin_preview_fps.setRange(5, 60)

        # Load persisted values or use sensible defaults
        width = int(self._settings.value("preview/width", 480))
        height = int(self._settings.value("preview/height", 270))
        fps = int(self._settings.value("preview/fps", 20))

        self.spin_preview_width.setValue(width)
        self.spin_preview_height.setValue(height)
        self.spin_preview_fps.setValue(fps)

        form.addRow("Preview width (px):", self.spin_preview_width)
        form.addRow("Preview height (px):", self.spin_preview_height)
        form.addRow("Preview FPS:", self.spin_preview_fps)

        info = QLabel(
            "These values are used as defaults by all visualization previews.\n"
            "You can still override them from the 3D visualizations tab for this session.",
            preview_group,
        )
        info.setWordWrap(True)
        form.addRow(info)

        preview_group.setLayout(form)
        layout.addWidget(preview_group)

        layout.addStretch(1)

        # Persist when changed
        self.spin_preview_width.valueChanged.connect(self._save_preview_settings)
        self.spin_preview_height.valueChanged.connect(self._save_preview_settings)
        self.spin_preview_fps.valueChanged.connect(self._save_preview_settings)
        self.combo_date_format.currentIndexChanged.connect(self._save_ui_settings)

        # Notify the main window when visual theme / scale / volume change
        self.combo_theme.currentIndexChanged.connect(self._on_theme_changed)
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        self.volume_slider.valueChanged.connect(self._on_volume_slider_changed)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def _save_preview_settings(self) -> None:
        """Persist preview defaults in QSettings."""
        self._settings.setValue("preview/width", int(self.spin_preview_width.value()))
        self._settings.setValue("preview/height", int(self.spin_preview_height.value()))
        self._settings.setValue("preview/fps", int(self.spin_preview_fps.value()))

    def _save_ui_settings(self) -> None:
        """Persist UI-related settings (date/time format)."""
        idx = self.combo_date_format.currentIndex()
        fmt = self.combo_date_format.itemData(idx)
        if not fmt:
            fmt = "%Y-%m-%d %H:%M"
        self._settings.setValue("ui/date_format", fmt)

    def _on_theme_changed(self) -> None:
        """Handle changes in the selected UI theme."""
        idx = self.combo_theme.currentIndex()
        theme_key = self.combo_theme.itemData(idx) or "neon"
        self._settings.setValue("ui/theme", theme_key)
        self.themeChanged.emit(str(theme_key))

    def _on_scale_changed(self) -> None:
        """Handle changes in the global UI scale percentage."""
        idx = self.combo_scale.currentIndex()
        value = self.combo_scale.itemData(idx)
        try:
            scale = int(value)
        except Exception:
            scale = 100
        self._settings.setValue("ui/scale_percent", scale)
        self.scaleChanged.emit(scale)

    def _on_volume_slider_changed(self, value: int) -> None:
        """
        Persist global volume in QSettings and notify listeners.

        The main window is expected to connect this signal and apply the
        volume to its shared QAudioOutput instance.
        """
        self.lbl_volume_value.setText(f"{value} %")
        self._settings.setValue("audio/volume_percent", int(value))
        self.volumeChanged.emit(int(value))
