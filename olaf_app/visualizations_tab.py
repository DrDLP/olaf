from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import soundfile as sf

from PyQt6.QtCore import Qt, QTimer, QUrl, QSettings
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QFormLayout,
    QHBoxLayout,
    QScrollArea,
    QCheckBox,
    QSlider,
    QMessageBox,
    QSpinBox,
)

from .project_manager import Project
from .visualizations_manager import VisualizationManager, VisualizationPluginInfo
from .visualization_api import PluginParameter

if TYPE_CHECKING:
    from .visualization_api import BaseVisualization


class VisualizationsTab(QWidget):
    """
    Main tab for audio-reactive 3D visualizations.

    It provides two sub-tabs:
      1) Plugin setup: select a visualization plugin and see its metadata.
      2) Preview / parameters: stem routing, parameter sliders, preview widget.
    """

    def __init__(self, player: QMediaPlayer, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.player = player
        self._project: Optional[Project] = None

        self._manager = VisualizationManager()

        self._current_plugin_id: Optional[str] = None
        self._plugin_instance: Optional["BaseVisualization"] = None
        self._preview_widget: Optional[QWidget] = None

        # Audio feature buffers for preview
        self._input_rms: Dict[str, np.ndarray] = {}
        self._rms_hop_ms: int = 50  # default, will be updated from preview FPS

        # Preview configuration (resolution + FPS)
        self._preview_width: int = 480
        self._preview_height: int = 270
        self._preview_fps: int = 20
        self._load_global_preview_settings()

        # Additional preview offset (ms) to compensate audio latency
        # Positive values make the visuals slightly earlier than the sound.
        self._preview_offset_ms: int = 0

        # --- UI references (assigned in _build_ui) ---
        self.subtabs: QTabWidget
        self.plugin_combo: QComboBox
        self.plugin_details_label: QLabel
        self.project_label: QLabel

        self.stem_form_layout: QFormLayout
        self.input_selectors: List[QComboBox] = []

        self.preview_info_label: QLabel
        self.parameters_group: QGroupBox
        self.parameter_form_layout: QFormLayout
        self.parameter_widgets: Dict[str, QWidget] = {}

        self.preview_group: QGroupBox
        self.preview_layout: QVBoxLayout
        self.preview_placeholder: QLabel

        self.preview_play_button: QPushButton
        self.preview_stop_button: QPushButton
        self._preview_playing: bool = False

        self.spin_tab_preview_width: QSpinBox
        self.spin_tab_preview_height: QSpinBox
        self.spin_tab_preview_fps: QSpinBox
        self.spin_tab_preview_offset: QSpinBox
        self.btn_preview_reset_defaults: QPushButton

        # Build UI once
        self._build_ui()

        # Timer driving audio features during preview.
        # It samples the QMediaPlayer position and uses precomputed RMS values.
        self._features_timer = QTimer(self)
        self._features_timer.setInterval(int(1000 / max(1, self._preview_fps)))
        self._features_timer.timeout.connect(self._on_features_tick)
        self._features_timer.start()

        # Initial plugin discovery / selection
        self.refresh_plugins()

    # ------------------------------------------------------------------ #
    # Public API (called from MainWindow)
    # ------------------------------------------------------------------ #

    def set_project(self, project: Optional[Project]) -> None:
        """
        Set the current project so we can list available stems
        for routing to plugin inputs.
        """
        self._project = project

        if project is None:
            self.project_label.setText("Current project: (none)")
        else:
            self.project_label.setText(f"Current project: {project.name}")

        self._populate_stem_choices()
        self._load_project_visualization_state()

    def refresh_plugins(self) -> None:
        """Rescan the visuals directory and update the plugin list."""
        self._manager.discover_plugins()

        self.plugin_combo.blockSignals(True)
        self.plugin_combo.clear()

        plugins = self._manager.list_plugins()
        if not plugins:
            self.plugin_combo.addItem("No plugins found", "")
            self.plugin_combo.setEnabled(False)
            self.plugin_details_label.setText(
                "No visualization plugins were found in the 'visuals' folder."
            )
            self._current_plugin_id = None
            self._instantiate_plugin(None)
            self._rebuild_stem_routing(None)
            self._rebuild_parameter_controls(None)
        else:
            self.plugin_combo.setEnabled(True)
            for info in plugins:
                self.plugin_combo.addItem(info.name, info.plugin_id)

            self.plugin_combo.blockSignals(False)
            # Select the first plugin by default
            self._on_plugin_selected(self.plugin_combo.currentIndex())
            return

        self.plugin_combo.blockSignals(False)

    def _load_global_preview_settings(self) -> None:
        """
        Load default preview resolution and FPS from QSettings.
        These values are used as initial configuration for the tab.
        """
        settings = QSettings("Olaf", "OlafApp")
        width = settings.value("preview/width", 480, type=int)
        height = settings.value("preview/height", 270, type=int)
        fps = settings.value("preview/fps", 20, type=int)

        self._preview_width = max(160, int(width))
        self._preview_height = max(90, int(height))
        self._preview_fps = max(5, min(60, int(fps)))

    def _restore_state_for_active_plugin(self, info: Optional[VisualizationPluginInfo]) -> None:
        """
        Restore parameters and routing for the currently selected plugin
        from the Project.visualizations mapping (if present).
        """
        if self._project is None or self._current_plugin_id is None:
            # Nothing to restore, just build controls from defaults
            if info is not None:
                self._rebuild_parameter_controls(info)
            return

        plugin_id = self._current_plugin_id
        visualizations = getattr(self._project, "visualizations", {}) or {}

        state = visualizations.get(plugin_id, {})
        params = state.get("parameters") or {}
        routing = state.get("routing") or {}

        # Apply parameters into the plugin instance
        if self._plugin_instance is not None and params:
            try:
                self._plugin_instance.config.update(params)
            except Exception:
                pass

        # Rebuild parameter widgets from updated config
        if info is not None:
            self._rebuild_parameter_controls(info)

        # Apply routing to combos
        if routing:
            for index, combo in enumerate(self.input_selectors):
                input_key = f"input_{index + 1}"
                stem_key = routing.get(input_key)
                if not stem_key:
                    continue
                idx = combo.findData(stem_key)
                if idx != -1:
                    combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        """
        Build the main 3D visualization UI as a single page with two columns.

        Left column (top to bottom):
            - Visualization plugin selector
            - Plugin description
            - Visualization preview (preview settings + widget + play/stop)
        Right column (top to bottom):
            - Plugin parameters
            - Stem routing
        """
        main_layout = QVBoxLayout(self)

        # ------------------------------------------------------------------
        # Two-column central layout
        # ------------------------------------------------------------------
        center_layout = QHBoxLayout()
        main_layout.addLayout(center_layout, stretch=1)

        # ===================== LEFT COLUMN =====================
        left_col = QVBoxLayout()
        center_layout.addLayout(left_col, stretch=2)

        # Project label kept but hidden (project name is shown in bottom bar)
        self.project_label = QLabel("", self)
        self.project_label.setVisible(False)
        left_col.addWidget(self.project_label)

        # --- Visualization plugin selector (top-left) ---------------------
        plugin_group = QGroupBox("Visualization plugin", self)
        plugin_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        plugin_layout = QHBoxLayout(plugin_group)

        plugin_layout.addWidget(QLabel("Plugin:", plugin_group))

        self.plugin_combo = QComboBox(plugin_group)
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_selected)
        plugin_layout.addWidget(self.plugin_combo, stretch=1)

        refresh_btn = QPushButton("Refresh plugins", plugin_group)
        refresh_btn.clicked.connect(self.refresh_plugins)
        plugin_layout.addWidget(refresh_btn)

        plugin_group.setLayout(plugin_layout)
        left_col.addWidget(plugin_group)

        # --- Plugin description (under selector) -------------------------
        details_group = QGroupBox("Plugin description", self)
        details_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        details_layout = QVBoxLayout(details_group)

        self.plugin_details_label = QLabel("", details_group)
        self.plugin_details_label.setWordWrap(True)
        details_layout.addWidget(self.plugin_details_label)

        details_group.setLayout(details_layout)
        left_col.addWidget(details_group)

        # --- Visualization preview (bottom-left) -------------------------
        self.preview_group = QGroupBox("Visualization preview", self)
        self.preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.preview_layout = QVBoxLayout(self.preview_group)

        preview_info = QLabel(
            "Real-time preview runs at reduced resolution and quality so it stays "
            "responsive. Final rendered videos will look much smoother and more "
            "detailed.",
            self.preview_group,
        )
        preview_info.setWordWrap(True)
        self.preview_layout.addWidget(preview_info)

        # Row with preview size / FPS / offset + reset button
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Preview size:", self.preview_group))

        self.spin_tab_preview_width = QSpinBox(self.preview_group)
        self.spin_tab_preview_width.setRange(160, 1920)
        self.spin_tab_preview_width.setValue(self._preview_width)
        size_row.addWidget(self.spin_tab_preview_width)

        size_row.addWidget(QLabel("×", self.preview_group))

        self.spin_tab_preview_height = QSpinBox(self.preview_group)
        self.spin_tab_preview_height.setRange(90, 1080)
        self.spin_tab_preview_height.setValue(self._preview_height)
        size_row.addWidget(self.spin_tab_preview_height)

        size_row.addSpacing(16)
        size_row.addWidget(QLabel("FPS:", self.preview_group))

        self.spin_tab_preview_fps = QSpinBox(self.preview_group)
        self.spin_tab_preview_fps.setRange(5, 60)
        self.spin_tab_preview_fps.setValue(self._preview_fps)
        size_row.addWidget(self.spin_tab_preview_fps)

        size_row.addSpacing(16)
        size_row.addWidget(QLabel("Offset (ms):", self.preview_group))

        self.spin_tab_preview_offset = QSpinBox(self.preview_group)
        self.spin_tab_preview_offset.setRange(-500, 500)
        self.spin_tab_preview_offset.setValue(self._preview_offset_ms)
        size_row.addWidget(self.spin_tab_preview_offset)

        size_row.addStretch(1)

        # The reset button now restores plugin parameters to their defaults.
        self.btn_preview_reset_defaults = QPushButton(
            "Reset to default",
            self.preview_group,
        )
        size_row.addWidget(self.btn_preview_reset_defaults)

        self.preview_layout.addLayout(size_row)

        offset_help = QLabel(
            "Positive values for the offset settings make visuals slightly earlier\n"
            "than the audio to compensate for playback latency.",
            self.preview_group,
        )
        offset_help.setWordWrap(True)
        self.preview_layout.addWidget(offset_help)

        # Preview widget / placeholder
        self.preview_placeholder = QLabel(
            "Visualization preview widget will be displayed here.",
            self.preview_group,
        )
        self.preview_placeholder.setFixedSize(self._preview_width, self._preview_height)
        self.preview_placeholder.setStyleSheet(
            "border: 1px dashed #555; background-color: #111; color: #777;"
        )        
        self.preview_placeholder.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.preview_layout.addWidget(
            self.preview_placeholder,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )


        # --- Play / Stop controls INSIDE the preview group ---------------
        controls_row = QHBoxLayout()
        self.preview_play_button = QPushButton("Play preview", self.preview_group)
        self.preview_stop_button = QPushButton("Stop", self.preview_group)

        self.preview_play_button.clicked.connect(self._on_preview_play_clicked)
        self.preview_stop_button.clicked.connect(self._on_preview_stop_clicked)

        controls_row.addWidget(self.preview_play_button)
        controls_row.addWidget(self.preview_stop_button)
        controls_row.addStretch(1)
        self.preview_layout.addLayout(controls_row)

        self.preview_group.setLayout(self.preview_layout)
        left_col.addWidget(self.preview_group)
        left_col.addStretch(1)

        # ===================== RIGHT COLUMN =====================
        right_col = QVBoxLayout()
        center_layout.addLayout(right_col, stretch=1)

        # --- Plugin parameters (top-right) -------------------------------
        self.parameters_group = QGroupBox("Plugin parameters", self)
        self.parameters_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.parameter_form_layout = QFormLayout(self.parameters_group)
        self.parameters_group.setLayout(self.parameter_form_layout)
        right_col.addWidget(self.parameters_group, stretch=1)

        # --- Stem routing (under parameters) -----------------------------
        routing_group = QGroupBox("Stem routing", self)
        routing_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.stem_form_layout = QFormLayout(routing_group)
        routing_group.setLayout(self.stem_form_layout)
        right_col.addWidget(routing_group)

        right_col.addStretch(1)

        # ------------------------------------------------------------------
        # Wire preview configuration controls to their handlers
        # ------------------------------------------------------------------
        self.spin_tab_preview_width.valueChanged.connect(self._on_preview_size_changed)
        self.spin_tab_preview_height.valueChanged.connect(self._on_preview_size_changed)
        self.spin_tab_preview_fps.valueChanged.connect(self._on_preview_fps_changed)
        self.spin_tab_preview_offset.valueChanged.connect(self._on_preview_offset_changed)
        self.btn_preview_reset_defaults.clicked.connect(self._on_preview_reset_defaults)


    # ------------------------------------------------------------------ #
    # Plugin selection / instantiation / UI updates
    # ------------------------------------------------------------------ #

    def _on_plugin_selected(self, index: int) -> None:
        """
        React to a change in the plugin combo box.

        This:
        - stores the current plugin id,
        - updates the description block,
        - instantiates the plugin and its preview widget,
        - rebuilds routing and parameter controls,
        - restores any saved state for this plugin in the current project.
        """
        plugin_id = self.plugin_combo.itemData(index)
        if not plugin_id:
            self._current_plugin_id = None
            if hasattr(self, "plugin_details_label"):
                self.plugin_details_label.setText("No plugin selected.")
            self._instantiate_plugin(None)
            self._rebuild_stem_routing(None)
            self._rebuild_parameter_controls(None)
            return

        info = self._manager.get_plugin(plugin_id)
        self._current_plugin_id = plugin_id

        if info is None:
            if hasattr(self, "plugin_details_label"):
                self.plugin_details_label.setText("Selected plugin metadata is not available.")
            self._instantiate_plugin(None)
            self._rebuild_stem_routing(None)
            self._rebuild_parameter_controls(None)
            return

        # Rich HTML description for the "Plugin description" block
        details = (
            f"<b>{info.name}</b> ({info.plugin_id})<br>"
            f"Author: {info.author} &nbsp;&nbsp; Version: {info.version}<br>"
            f"Max inputs: {info.max_inputs}<br><br>"
            f"{info.description or 'No description provided.'}"
        )
        if hasattr(self, "plugin_details_label"):
            self.plugin_details_label.setText(details)

        # Instantiate plugin + routing, then restore any saved state
        self._instantiate_plugin(info)
        self._rebuild_stem_routing(info)
        self._restore_state_for_active_plugin(info)

        # Save state immediately (creates the entry in project.visualizations if needed)
        self._save_visualization_to_project()

    def _instantiate_plugin(self, info: Optional[VisualizationPluginInfo]) -> None:
        """
        Create or destroy the plugin instance and its preview widget
        according to the selected plugin.
        """
        # Deactivate and drop previous instance
        if self._plugin_instance is not None:
            try:
                self._plugin_instance.on_deactivate()
            except Exception:
                pass
        self._plugin_instance = None

        # Clear previous preview widget
        if self._preview_widget is not None:
            self.preview_layout.removeWidget(self._preview_widget)
            self._preview_widget.deleteLater()
            self._preview_widget = None

        if info is None:
            self.preview_placeholder.setVisible(True)
            return

        instance = self._manager.create_instance(info.plugin_id, config=None)
        if instance is None:
            self.preview_placeholder.setVisible(True)
            return

        self._plugin_instance = instance

        # Create and attach the plugin's preview widget
        try:
            widget = instance.create_widget(self.preview_group)
        except Exception:
            self._plugin_instance = None
            self.preview_placeholder.setVisible(True)
            return

        self._preview_widget = widget
        self.preview_layout.addWidget(
            widget,
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self.preview_placeholder.setVisible(False)


        # Apply current preview settings to the new widget / plugin
        self._apply_preview_settings_to_plugin()

        try:
            instance.on_activate()
        except Exception:
            pass

        # Reset feature buffers when plugin changes
        self._input_rms.clear()

    def _rebuild_stem_routing(self, info: Optional[VisualizationPluginInfo]) -> None:
        """
        Rebuild the stem routing form rows based on the selected plugin's
        maximum number of inputs (capped at 5).
        """
        # Clear existing rows
        while self.stem_form_layout.count():
            item = self.stem_form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.input_selectors = []

        if info is None:
            return

        # Number of inputs defined by plugin (hard-limited to 5 for UI sanity)
        n_inputs = max(1, min(info.max_inputs, 5))
        for i in range(n_inputs):
            label = QLabel(f"Input {i + 1}:", self)
            combo = QComboBox(self)
            combo.currentIndexChanged.connect(self._on_routing_changed)
            self.stem_form_layout.addRow(label, combo)
            self.input_selectors.append(combo)

        self._populate_stem_choices()

    def _on_routing_changed(self, index: int) -> None:
        # We do not care which combo was changed, just save the whole routing
        self._save_visualization_to_project()

    def _rebuild_parameter_controls(self, info: Optional[VisualizationPluginInfo]) -> None:
        """
        Rebuild the parameter widgets list according to the plugin's
        declared parameters.
        """
        # Clear existing widgets
        while self.parameter_form_layout.count():
            item = self.parameter_form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.parameter_widgets.clear()

        if info is None or not info.parameters:
            placeholder = QLabel(
                "This plugin does not declare any configurable parameters.",
                self,
            )
            self.parameter_form_layout.addRow(placeholder)
            return

        plugin_config: Dict[str, Any] = {}
        if self._plugin_instance is not None and info.plugin_id == self._current_plugin_id:
            plugin_config = getattr(self._plugin_instance, "config", {}) or {}

        for name, param in info.parameters.items():
            current_value = plugin_config.get(name, param.default)
            widget = self._create_widget_for_parameter(param, current_value)
            self.parameter_widgets[name] = widget
            label = QLabel(param.label or name, self.parameters_group)
            self.parameter_form_layout.addRow(label, widget)
            self._connect_parameter_widget(name, param, widget)

    # ------------------------------------------------------------------ #
    # Preview configuration (size + FPS)
    # ------------------------------------------------------------------ #

    def _apply_preview_settings_to_plugin(self) -> None:
        """
        Apply the current preview resolution / FPS to the active plugin
        and its widget, and update the feature timer.
        """
        # Update widget size if present
        if self._preview_widget is not None:
            self._preview_widget.setMinimumSize(self._preview_width, self._preview_height)
            self._preview_widget.setMaximumSize(self._preview_width, self._preview_height)

        # Notify plugin instance (optional hook)
        if self._plugin_instance is not None:
            try:
                self._plugin_instance.apply_preview_settings(
                    self._preview_width,
                    self._preview_height,
                    self._preview_fps,
                )
            except Exception:
                # Plugins are allowed to ignore or mishandle this hook
                pass

        # Keep feature timer and RMS step roughly aligned with visual FPS
        fps = max(1, int(self._preview_fps))
        interval_ms = max(10, int(1000 / fps))
        self._features_timer.setInterval(interval_ms)
        self._rms_hop_ms = interval_ms

    def _on_preview_size_changed(self) -> None:
        """Handle per-tab width / height edits."""
        self._preview_width = int(self.spin_tab_preview_width.value())
        self._preview_height = int(self.spin_tab_preview_height.value())
        self.preview_placeholder.setFixedSize(self._preview_width, self._preview_height)
        self._apply_preview_settings_to_plugin()

    def _on_preview_fps_changed(self, value: int) -> None:
        """Handle per-tab FPS edits."""
        self._preview_fps = int(value)
        self._apply_preview_settings_to_plugin()

    def _on_preview_offset_changed(self, value: int) -> None:
        """
        Handle per-tab visual offset edits.

        Positive values make visuals slightly ahead of the audio to
        compensate for playback latency.
        """
        self._preview_offset_ms = int(value)

    def _on_preview_reset_defaults(self) -> None:
        """
        Reset all plugin parameters to their declared default values.

        The button labelled "Reset to default" in the preview block uses this
        handler. It does **not** touch the preview size/FPS/offset settings;
        it only restores the active plugin configuration.
        """
        if not self._current_plugin_id:
            return

        info = self._manager.get_plugin(self._current_plugin_id)
        if info is None or not info.parameters:
            return

        if self._plugin_instance is None:
            return

        # Ensure the plugin instance has a config dict
        if getattr(self._plugin_instance, "config", None) is None:
            self._plugin_instance.config = {}

        # IMPORTANT: iterate over items(), not over keys
        for name, param in info.parameters.items():
            widget = self.parameter_widgets.get(name)
            if widget is None:
                continue

            default_value = param.default

            # Bool parameter -> QCheckBox
            if isinstance(widget, QCheckBox):
                value = bool(default_value)
                widget.blockSignals(True)
                widget.setChecked(value)
                widget.blockSignals(False)
                self._on_parameter_changed(name, value)
                continue

            # Enum parameter -> QComboBox
            if isinstance(widget, QComboBox):
                target_index = -1
                for i in range(widget.count()):
                    if widget.itemData(i) == default_value:
                        target_index = i
                        break
                if target_index == -1 and widget.count() > 0:
                    target_index = 0

                if target_index != -1:
                    widget.blockSignals(True)
                    widget.setCurrentIndex(target_index)
                    widget.blockSignals(False)
                    value = widget.itemData(target_index)
                    self._on_parameter_changed(name, value)
                continue

            # Numeric parameter -> slider container
            slider = getattr(widget, "_slider", None)
            value_label = getattr(widget, "_value_label", None)
            param_type = getattr(widget, "_param_type", None)

            if not isinstance(slider, QSlider):
                continue

            if param_type == "int":
                min_v = slider.minimum()
                max_v = slider.maximum()
                try:
                    v = int(default_value)
                except Exception:
                    v = min_v
                v = max(min_v, min(max_v, v))

                slider.blockSignals(True)
                slider.setValue(v)
                slider.blockSignals(False)

                if isinstance(value_label, QLabel):
                    value_label.setText(str(v))

                self._on_parameter_changed(name, v)

            elif param_type == "float":
                min_v = float(getattr(widget, "_min", 0.0))
                step = float(getattr(widget, "_step", 0.01))
                max_index = slider.maximum()

                try:
                    v = float(default_value)
                except Exception:
                    v = min_v

                # Clamp within representable range
                max_v = min_v + step * max_index
                if max_v < min_v:
                    max_v = min_v
                v = max(min_v, min(max_v, v))

                index = int(round((v - min_v) / step)) if step > 0 else 0
                index = max(0, min(max_index, index))

                slider.blockSignals(True)
                slider.setValue(index)
                slider.blockSignals(False)

                if isinstance(value_label, QLabel):
                    value_label.setText(f"{v:.3g}")

                self._on_parameter_changed(name, v)

    # ------------------------------------------------------------------ #
    # Persistence: plugin selection / params / routing in Project
    # ------------------------------------------------------------------ #

    def _save_visualization_to_project(self) -> None:
        """Persist current visualization selection and settings into the project."""
        if self._project is None:
            return
        if not self._current_plugin_id:
            return

        plugin_id = self._current_plugin_id

        # Ensure the dict exists on the Project
        if getattr(self._project, "visualizations", None) is None:
            self._project.visualizations = {}

        # Plugin parameters
        if self._plugin_instance is not None:
            config = getattr(self._plugin_instance, "config", {}) or {}
            params = dict(config)
        else:
            params = {}

        # Stem routing (input_1, input_2, ...)
        routing: Dict[str, str] = {}
        for index, combo in enumerate(self.input_selectors):
            stem_key = combo.currentData()
            if stem_key:
                routing[f"input_{index + 1}"] = stem_key

        # Store per-plugin state
        self._project.visualizations[plugin_id] = {
            "parameters": params,
            "routing": routing,
        }

        # Remember last-used plugin
        self._project.visualization_plugin_id = plugin_id

        # Save project metadata
        try:
            self._project.save()
        except Exception:
            # Never crash the UI if saving fails
            pass


    def _load_project_visualization_state(self) -> None:
        """
        When a new project is set, restore which plugin was last used,
        then let _on_plugin_selected + _restore_state_for_active_plugin
        re-apply its parameters and routing.
        """
        if self._project is None:
            return

        visualizations = getattr(self._project, "visualizations", {}) or {}
        last_plugin_id = getattr(self._project, "visualization_plugin_id", None)

        target_plugin_id: Optional[str] = None

        # 1) If a last-used plugin exists and has a saved state, use it.
        if last_plugin_id and last_plugin_id in visualizations:
            target_plugin_id = last_plugin_id
        # 2) Otherwise, if at least one plugin has a state, use the first one.
        elif visualizations:
            target_plugin_id = next(iter(visualizations.keys()))

        if not target_plugin_id:
            return

        # Select this plugin in the combo (this will trigger _on_plugin_selected)
        idx = self.plugin_combo.findData(target_plugin_id)
        if idx != -1:
            self.plugin_combo.setCurrentIndex(idx)


    # ------------------------------------------------------------------ #
    # Parameter widgets helpers (sliders)
    # ------------------------------------------------------------------ #

    def _create_widget_for_parameter(
        self,
        param: PluginParameter,
        current_value: Any,
    ) -> QWidget:
        """
        Create a Qt widget for a given PluginParameter and initial value.

        Numeric parameters (int / float) are controlled by sliders; bool
        parameters use a checkbox; enum parameters use a combo box.
        """
        # Bool parameter
        if param.type == "bool":
            w = QCheckBox(self.parameters_group)
            try:
                w.setChecked(bool(current_value))
            except Exception:
                w.setChecked(bool(param.default))
            return w

        # Enum parameter
        if param.type == "enum" and param.choices:
            w = QComboBox(self.parameters_group)
            for choice in param.choices:
                w.addItem(str(choice), choice)

            index_to_select = 0
            for i in range(w.count()):
                if w.itemData(i) == current_value:
                    index_to_select = i
                    break
            w.setCurrentIndex(index_to_select)
            return w

        # Numeric parameters: int / float -> slider + value label
        container = QWidget(self.parameters_group)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Orientation.Horizontal, container)
        slider.setTickPosition(QSlider.TickPosition.NoTicks)

        value_label = QLabel(container)
        value_label.setMinimumWidth(60)

        # Attach for later retrieval
        setattr(container, "_slider", slider)
        setattr(container, "_value_label", value_label)

        if param.type == "int":
            # Boundaries
            min_v = int(param.minimum if param.minimum is not None else 0)
            max_v = int(param.maximum if param.maximum is not None else 100)
            if max_v < min_v:
                max_v = min_v

            step = int(param.step if param.step is not None else 1)
            if step <= 0:
                step = 1

            slider.setMinimum(min_v)
            slider.setMaximum(max_v)
            slider.setSingleStep(step)

            try:
                value = int(current_value)
            except Exception:
                value = int(param.default)

            value = max(min_v, min(max_v, value))
            slider.setValue(value)
            value_label.setText(str(value))

            setattr(container, "_param_type", "int")

        elif param.type == "float":
            # Use an integer slider that indexes discrete float steps
            min_v = float(param.minimum if param.minimum is not None else 0.0)
            max_v = float(param.maximum if param.maximum is not None else 1.0)
            if max_v < min_v:
                max_v = min_v + 1.0

            # Step defined by the plugin or derived from the range
            if param.step is not None and param.step > 0.0:
                step = float(param.step)
            else:
                step = (max_v - min_v) / 100.0 if max_v > min_v else 0.01

            if step <= 0.0:
                step = 0.01

            num_steps = max(1, int(round((max_v - min_v) / step)))
            slider.setMinimum(0)
            slider.setMaximum(num_steps)

            try:
                value = float(current_value)
            except Exception:
                value = float(param.default)

            value = max(min_v, min(max_v, value))
            index = int(round((value - min_v) / step))
            index = max(0, min(num_steps, index))
            slider.setValue(index)

            value_label.setText(f"{value:.3g}")

            setattr(container, "_param_type", "float")
            setattr(container, "_min", min_v)
            setattr(container, "_step", step)

        else:
            # Fallback: unsupported parameter type
            return QLabel(f"(Unsupported parameter type: {param.type})", self.parameters_group)

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
            def _on_index_changed(
                index: int,
                combo: QComboBox = widget,
                param_name: str = name,
            ) -> None:
                value = combo.itemData(index)
                self._on_parameter_changed(param_name, value)

            widget.currentIndexChanged.connect(_on_index_changed)
            return

        # Slider-based numeric parameter
        slider = getattr(widget, "_slider", None)
        if isinstance(slider, QSlider):
            slider.valueChanged.connect(
                lambda idx, n=name, w=widget: self._on_slider_moved(n, w, idx)
            )

    def _on_slider_moved(self, name: str, container: QWidget, index: int) -> None:
        """
        Called when a numeric parameter slider is moved.
        Translates slider position into a numeric value and updates config.
        """
        param_type = getattr(container, "_param_type", "int")
        value_label: Optional[QLabel] = getattr(container, "_value_label", None)

        if param_type == "int":
            value = int(index)
            if value_label is not None:
                value_label.setText(str(value))
        else:
            # float
            min_v = float(getattr(container, "_min", 0.0))
            step = float(getattr(container, "_step", 1.0))
            value = min_v + index * step
            if value_label is not None:
                value_label.setText(f"{value:.3g}")

        self._on_parameter_changed(name, value)

    def _on_parameter_changed(self, name: str, value: Any) -> None:
        """
        Update the configuration dictionary of the active plugin instance
        when a parameter widget changes.
        """
        if self._plugin_instance is None:
            return

        try:
            self._plugin_instance.config[name] = value
        except Exception:
            # Do not let plugins crash the host because of config handling.
            return

        self._save_visualization_to_project()

    # ------------------------------------------------------------------ #
    # Stems mapping helpers
    # ------------------------------------------------------------------ #

    def _populate_stem_choices(self) -> None:
        """
        Fill each input selector combo with available stems from the current
        project plus the full mix, if available.
        """
        stems = self._get_available_stems()

        for combo in self.input_selectors:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(Not connected)", "")
            for label, key in stems:
                combo.addItem(label, key)
            combo.blockSignals(False)

    def _get_available_stems(self) -> List[Tuple[str, str]]:
        """
        Build a list of (label, key) for available stems.

        The key will be stored later in the routing configuration and may look like:
            "full_mix"
            "htdemucs:vocals"
            "htdemucs_6s:guitar"
        """
        result: List[Tuple[str, str]] = []

        if self._project is None:
            return result

        # Full mix
        if self._project.audio_file:
            result.append(("Full mix (project audio)", "full_mix"))

        # Stems by model
        for model_name, stems_dict in self._project.stems_by_model.items():
            for stem_name in stems_dict.keys():
                label = f"{model_name}: {stem_name}"
                key = f"{model_name}:{stem_name}"
                result.append((label, key))

        return result

    def _resolve_stem_path(self, key: str) -> Optional[Path]:
        """
        Translate a routing key into an absolute Path to an audio file.

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

    # ------------------------------------------------------------------ #
    # Preview controls + audio mixing
    # ------------------------------------------------------------------ #

    def _on_preview_play_clicked(self) -> None:
        """
        Build a temporary mix of the selected stems, compute RMS features,
        then play that mix through the shared QMediaPlayer while sending
        features to the active visualization plugin.
        """
        if self._project is None:
            QMessageBox.warning(self, "No project selected", "Please select a project first.")
            return

        # Collect selected stem paths, mapped by input key
        input_paths: Dict[str, Path] = {}
        for index, combo in enumerate(self.input_selectors):
            stem_key = combo.currentData()
            if not stem_key:
                continue
            path = self._resolve_stem_path(stem_key)
            if path is not None and path.exists():
                input_paths[f"input_{index + 1}"] = path

        if not input_paths:
            QMessageBox.warning(
                self,
                "No stems selected",
                "Please assign at least one input to a stem or to the full mix.",
            )
            return

        try:
            mix_path = self._build_preview_mix_and_features(input_paths)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Preview error",
                f"Could not prepare preview audio:\n{e}",
            )
            return

        # Start playback
        url = QUrl.fromLocalFile(str(mix_path.resolve()))
        self.player.setSource(url)
        self.player.setPosition(0)
        self.player.play()

        self._preview_playing = True

    def _on_preview_stop_clicked(self) -> None:
        """Stop the preview playback."""
        self._preview_playing = False
        self.player.stop()

    def _build_preview_mix_and_features(self, input_paths: Dict[str, Path]) -> Path:
        """
        Load all selected stems, compute RMS envelopes for each input, and
        build a summed stereo mix written to the project folder as a
        temporary WAV file. Returns the path to that file.

        This function assumes that all stems share the same sample rate.
        """
        # Determine output path inside the project folder
        assert self._project is not None
        out_path = self._project.folder / "preview_mix.wav"

        self._input_rms.clear()

        sample_rate: Optional[int] = None
        mix_data: Optional[np.ndarray] = None

        hop_ms = self._rms_hop_ms

        for input_key, path in input_paths.items():
            data, sr = sf.read(path, always_2d=True)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                # Simple resampling using ratio, but in practice stems should all share the same sr.
                factor = sample_rate / sr
                new_length = int(data.shape[0] * factor)
                indices = (np.arange(new_length) / factor).astype(np.int64)
                indices = np.clip(indices, 0, data.shape[0] - 1)
                data = data[indices]

            # Ensure float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Mixdown to stereo: if 1 channel, duplicate; if >2, take first 2.
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]

            # Accumulate into mix_data
            if mix_data is None:
                mix_data = data
            else:
                min_len = min(mix_data.shape[0], data.shape[0])
                mix_data = mix_data[:min_len]
                mix_data += data[:min_len]

            # Compute RMS envelope for this input
            mono = data.mean(axis=1)  # simple downmix
            hop_samples = max(1, int(sample_rate * (hop_ms / 1000.0)))
            rms_values: List[float] = []
            for start in range(0, mono.shape[0], hop_samples):
                chunk = mono[start : start + hop_samples]
                if chunk.size == 0:
                    break
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                # Normalize moderately assuming [-1, 1] amplitude
                rms_values.append(min(1.0, rms * 2.0))

            self._input_rms[input_key] = np.array(rms_values, dtype=np.float32)

        if mix_data is None or sample_rate is None:
            raise RuntimeError("No valid stem data could be loaded.")

        # Normalize mix to avoid clipping
        peak = float(np.max(np.abs(mix_data)))
        if peak > 1.0:
            mix_data /= peak

        # Write the mix to disk
        sf.write(out_path, mix_data, sample_rate)

        return out_path

    def _on_features_tick(self) -> None:
        """
        Called regularly to feed audio features to the active plugin
        while the preview is playing.
        """
        if self._plugin_instance is None:
            return

        if not self._preview_playing:
            return

        raw_ms = int(self.player.position())
        if raw_ms < 0:
            return

        # Apply visual offset to compensate audio latency
        effective_ms = raw_ms + self._preview_offset_ms
        if effective_ms < 0:
            effective_ms = 0

        features_inputs: Dict[str, Dict[str, float]] = {}

        for index, combo in enumerate(self.input_selectors):
            input_key = f"input_{index + 1}"
            stem_key = combo.currentData()
            if not stem_key:
                # Not connected
                features_inputs[input_key] = {"rms": 0.0}
                continue

            rms_array = self._input_rms.get(input_key)
            if rms_array is None or rms_array.size == 0:
                features_inputs[input_key] = {"rms": 0.0}
                continue

            frame_index = int(effective_ms / self._rms_hop_ms)
            if frame_index < 0 or frame_index >= rms_array.size:
                level = 0.0
            else:
                level = float(rms_array[frame_index])

            features_inputs[input_key] = {"rms": level}

        features: Dict[str, Any] = {
            "time_ms": effective_ms,
            "inputs": features_inputs,
        }

        try:
            self._plugin_instance.on_audio_features(features)
        except Exception:
            # Do not let plugin errors crash the UI.
            pass
