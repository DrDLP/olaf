from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

import time
import copy

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import librosa  # type: ignore[import]
except Exception:  # pragma: no cover
    librosa = None  # type: ignore[assignment]

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QSizePolicy,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QTabWidget,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QSlider,
)

from .project_manager import Project
from .cover_visualizations_manager import CoverVisualizationsManager, CoverEffectInfo
from .cover_visualization_api import FrameFeatures
from .visualization_api import PluginParameter  # noqa: F401 (used for type hints in plugins)

@dataclass
class AudioEnvelope:
    """Precomputed RMS envelope used for live previews."""
    path: Path
    rms: np.ndarray
    duration: float
    fps: int


class CoverVisualizationsTab(QWidget):
    """
    Tab for 2D visualizations based on the project cover image.

    This tab mirrors the 3D Visualizations tab, but focused on image-based
    effects (glitch, vignette, breathing zoom, etc.).

    Responsibilities:
      - Display the current project and its cover image.
      - Discover 2D visualization plugins from a dedicated folder.
      - Let the user configure each effect (parameters + audio source).
      - Build an ordered effect chain (execution order).
      - Store everything into the Project (cover_visual_effects / cover_visual_chain).
      - Provide simple preview:
          * per-effect preview in "Plugin setup"
          * cumulative preview in "Preview / parameters" (applies full chain once).
    """

    COVER_VISUALS_ROOT = Path("cover_visuals")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._project: Optional[Project] = None

        # Plugin discovery / instantiation
        self._manager = CoverVisualizationsManager(visuals_root=self.COVER_VISUALS_ROOT)
        self._plugins_by_id: Dict[str, CoverEffectInfo] = {}

        # Ordered list of *instance keys* forming the current effect chain.
        # Each entry is either:
        #   - "effect_id"           (legacy format)
        #   - "effect_id#N"         (instance N of that effect).
        self._effect_chain: List[str] = []

        # Per-instance configuration in memory:
        #   instance_key -> {
        #       "effect_id": "...",
        #       "parameters": {...},
        #       "routing": {"audio_source": "..."}
        #   }
        #
        # Old projects may still use effect_id as keys. We handle both.
        self._effect_configs: Dict[str, Dict[str, Any]] = {}

        # Mapping param_name -> widget (for currently selected plugin)
        self._param_widgets: Dict[str, QWidget] = {}

        # Index in _effect_chain of the instance currently being edited,
        # or None if editing a "template" (no specific instance).
        self._current_chain_index: Optional[int] = None

        # Instance key (effect_id or effect_id#N) currently bound to the
        # parameter + routing UI. All saves go to this key.
        self._current_instance_key: Optional[str] = None

        # When True, UI updates (sliders, audio source) must not trigger saves.
        self._suspend_save: bool = False
        
        # Internal flag to distinguish plugin_combo changes coming from
        # a chain selection (we do NOT clear _current_chain_index then).
        self._plugin_combo_change_from_chain: bool = False

        # UI references (assigned in _build_ui)
        self.project_label: QLabel
        self.cover_label: QLabel
        self.plugin_combo: QComboBox
        self.plugin_details_label: QLabel
        self.chain_list: QListWidget
        self.btn_add_effect: QPushButton
        self.btn_remove_effect: QPushButton
        self.btn_move_up: QPushButton
        self.btn_move_down: QPushButton
        self.btn_rescan_plugins: QPushButton
        self.audio_source_combo: QComboBox
        self.btn_preview_effect: QPushButton
        self.chain_audio_source_combo: QComboBox
        self.btn_preview_chain_live: QPushButton
        self.params_form_layout: QFormLayout

        self._build_ui()
        # Populate audio source combos (including playback) even without project
        self._refresh_audio_source_combo()
        self._rescan_plugins()


    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        """
        Build the UI for 2D cover-based visualizations.

        Layout:

          - Left: cover preview (used for single-effect + full-chain previews).
          - Right (top to bottom):
              * Effect chain
              * Chain preview (playback source + live preview)
              * Available 2D effects
              * Effect parameters
              * Audio routing + single-effect preview
        """
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Title
        #title = QLabel("2D visualizations (cover-based)", self)
        #title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        #title.setStyleSheet("font-size: 16px; font-weight: bold;")
        #main_layout.addWidget(title)

        # Current project label (now hidden: shown globally in bottom bar)
        self.project_label = QLabel("", self)
        self.project_label.setVisible(False)
        main_layout.addWidget(self.project_label)

        # Main content: left (cover) / right (controls)
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(5, 5, 5, 5)
        content_layout.setSpacing(10)
        main_layout.addLayout(content_layout, stretch=1)

        # --------------------------------------------------------------
        # Left column: cover preview + 2D effects + routing
        # --------------------------------------------------------------
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(10)
        content_layout.addLayout(left_col, stretch=1)

        self.cover_label = QLabel("No cover loaded.", self)
        self.cover_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        self.cover_label.setMinimumSize(320, 320)
        self.cover_label.setStyleSheet(
            "background-color: #202020; color: #AAAAAA; border: 1px solid #404040;"
        )
        self.cover_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        left_col.addWidget(self.cover_label, stretch=1)


        # --------------------------------------------------------------
        # Right: all controls (chain, preview, plugins, params, routing)
        # --------------------------------------------------------------
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(10)
        content_layout.addLayout(right_col, stretch=1)

        # --- Group: Effect chain --------------------------------------
        chain_group = QGroupBox("Effect chain (execution order)", self)
        chain_layout = QVBoxLayout(chain_group)

        self.chain_list = QListWidget(chain_group)
        self.chain_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.chain_list.currentRowChanged.connect(self._on_chain_selection_changed)
        chain_layout.addWidget(self.chain_list)

        chain_buttons_row = QHBoxLayout()
        self.btn_add_effect = QPushButton("Add selected effect", chain_group)
        self.btn_remove_effect = QPushButton("Remove from chain", chain_group)
        self.btn_move_up = QPushButton("Move up", chain_group)
        self.btn_move_down = QPushButton("Move down", chain_group)

        self.btn_add_effect.clicked.connect(self._on_add_effect)
        self.btn_remove_effect.clicked.connect(self._on_remove_effect)
        self.btn_move_up.clicked.connect(self._on_move_up)
        self.btn_move_down.clicked.connect(self._on_move_down)

        chain_buttons_row.addWidget(self.btn_add_effect)
        chain_buttons_row.addWidget(self.btn_remove_effect)
        chain_buttons_row.addWidget(self.btn_move_up)
        chain_buttons_row.addWidget(self.btn_move_down)

        chain_layout.addLayout(chain_buttons_row)

        right_col.addWidget(chain_group)

        # --- Group: Chain preview (live) ------------------------------
        chain_preview_group = QGroupBox("Chain preview", self)
        cp_layout = QVBoxLayout(chain_preview_group)

        # Playback audio source for full-chain preview (what you HEAR)
        source_row = QHBoxLayout()
        source_label = QLabel("Playback audio source:", chain_preview_group)
        self.chain_audio_source_combo = QComboBox(chain_preview_group)
        source_row.addWidget(source_label)
        source_row.addWidget(self.chain_audio_source_combo)
        cp_layout.addLayout(source_row)

        self.btn_preview_chain_live = QPushButton(
            "Play audio with live full-chain preview", chain_preview_group
        )
        self.btn_preview_chain_live.clicked.connect(self._on_preview_chain_live)

        cp_layout.addWidget(self.btn_preview_chain_live)

        right_col.addWidget(chain_preview_group)

        # --- Group: Available effects ---------------------------------
        plugins_group = QGroupBox("Available 2D effects", self)
        plugins_layout = QFormLayout(plugins_group)

        self.plugin_combo = QComboBox(plugins_group)
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_combo_changed)

        self.btn_rescan_plugins = QPushButton("Rescan plugins", plugins_group)
        self.btn_rescan_plugins.clicked.connect(self._rescan_plugins)

        combo_row = QHBoxLayout()
        combo_row.addWidget(self.plugin_combo)
        combo_row.addWidget(self.btn_rescan_plugins)
        plugins_layout.addRow("Effect script:", combo_row)

        self.plugin_details_label = QLabel(
            "No 2D plugins found in 'cover_visuals/'.\n"
            "Add Python files defining BaseCoverEffect subclasses to enable effects.",
            plugins_group,
        )
        self.plugin_details_label.setWordWrap(True)
        plugins_layout.addRow(self.plugin_details_label)

        left_col.addWidget(plugins_group)

        # --- Group: Parameters for selected effect --------------------
        params_group = QGroupBox("Effect parameters", self)
        self.params_form_layout = QFormLayout(params_group)
        self.params_form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        right_col.addWidget(params_group)

        # --- Group: Audio routing + per-effect preview ----------------
        routing_group = QGroupBox("Audio routing & single-effect preview", self)
        routing_layout = QFormLayout(routing_group)

        self.audio_source_combo = QComboBox(routing_group)
        self.audio_source_combo.currentIndexChanged.connect(self._on_audio_source_changed)
        routing_layout.addRow("Audio source:", self.audio_source_combo)

        self.btn_preview_effect = QPushButton(
            "Preview this effect on cover", routing_group
        )
        self.btn_preview_effect.clicked.connect(self._on_preview_single_effect)
        routing_layout.addRow(self.btn_preview_effect)

        left_col.addWidget(routing_group)

        # Push everything up
        right_col.addStretch(1)


    def resizeEvent(self, event) -> None:
        """
        Ensure the cover preview always uses the full available width.

        Each time the tab is resized (including when it is first shown),
        we rescale the current pixmap to the new label size so that the
        image behaves like after a manual preview.
        """
        super().resizeEvent(event)
        pix = self.cover_label.pixmap()
        if pix is not None and not pix.isNull():
            # Rescale the current image (original cover or preview frame)
            self._set_pixmap_scaled(pix)


    # ------------------------------------------------------------------ #
    # Plugin discovery                                                   #
    # ------------------------------------------------------------------ #

    def _rescan_plugins(self) -> None:
        """
        Rescan the cover_visuals folder and repopulate the plugins combo.
        """
        self._manager.discover_plugins()

        self.plugin_combo.blockSignals(True)
        self.plugin_combo.clear()
        self._plugins_by_id.clear()
        self.plugin_combo.addItem("(no effect selected)", "")

        effects = self._manager.list_plugins()
        for info in effects:
            self._plugins_by_id[info.effect_id] = info
            self.plugin_combo.addItem(info.name, info.effect_id)

        self.plugin_combo.blockSignals(False)

        if not effects:
            self.plugin_details_label.setText(
                "No 2D plugins found in 'cover_visuals/'.\n"
                "Add Python files defining BaseCoverEffect subclasses to enable effects."
            )
        else:
            self.plugin_details_label.setText(
                "Select an effect from the combo box to configure its parameters."
            )

        # When plugins have changed, the chain may contain unknown IDs
        self._sanitize_chain_against_plugins()
        self._rebuild_chain_list()
        self._reload_current_plugin_ui()

    # ------------------------------------------------------------------ #
    # Instance key helpers                                               #
    # ------------------------------------------------------------------ #

    def _split_chain_entry(self, entry: str) -> tuple[str, Optional[int]]:
        """
        Split a chain entry into (effect_id, instance_index).

        Examples
        --------
        "olaf_cover_glitch"      -> ("olaf_cover_glitch", None)
        "olaf_cover_glitch#2"    -> ("olaf_cover_glitch", 2)
        """
        if "#" not in entry:
            return entry, None
        base, suffix = entry.rsplit("#", 1)
        try:
            idx = int(suffix)
        except ValueError:
            return entry, None
        return base, idx


    def _sanitize_chain_against_plugins(self) -> None:
        """
        Remove any chain entry whose underlying effect_id is no longer
        available in the discovered plugin list.

        Also keeps _effect_configs consistent for new-format instance keys.
        Old per-effect configs are kept as a fallback for legacy projects.
        """
        valid_ids = set(self._plugins_by_id.keys())
        new_chain: List[str] = []

        for entry in self._effect_chain:
            effect_id, _ = self._split_chain_entry(entry)
            if effect_id in valid_ids:
                new_chain.append(entry)

        self._effect_chain = new_chain

        # Keep only configs that are still referenced by the chain,
        # but do not delete legacy per-effect configs (keys == effect_id),
        # as they can still serve as default for old projects.
        referenced_keys = set(self._effect_chain)
        new_configs: Dict[str, Dict[str, Any]] = {}
        for key, cfg in self._effect_configs.items():
            if key in referenced_keys:
                new_configs[key] = cfg
            else:
                # Preserve legacy per-effect entries like "olaf_cover_glitch"
                eff_id, idx = self._split_chain_entry(key)
                if idx is None and eff_id in valid_ids:
                    new_configs[key] = cfg

        self._effect_configs = new_configs


    # ------------------------------------------------------------------ #
    # Plugin selection & parameter UI                                    #
    # ------------------------------------------------------------------ #

    def _on_plugin_combo_changed(self, index: int) -> None:
        """
        React to plugin combo changes.

        If the change comes from selecting an entry in the effect chain,
        we keep the current chain index and just rebuild the UI.
        Otherwise (user manually chose an effect in the combo), we detach
        from any specific instance and show a template UI.
        """
        if getattr(self, "_plugin_combo_change_from_chain", False):
            # Change triggered by chain selection: keep current_chain_index
            self._reload_current_plugin_ui()
        else:
            # User changed the plugin type manually → no specific chain binding
            self._current_chain_index = None
            self._reload_current_plugin_ui()

    def _reload_current_plugin_ui(self) -> None:
        """
        Rebuild the parameter widgets and routing combo.

        If a chain entry is selected and its effect_id matches the current
        plugin combo, we load parameters + routing from that instance's
        configuration in self._effect_configs.

        Otherwise, we show a "template" UI (no bound instance, not persisted).
        """
        # Clear old widgets
        for i in reversed(range(self.params_form_layout.count())):
            item = self.params_form_layout.itemAt(i)
            if item is not None and item.widget() is not None:
                w = item.widget()
                self.params_form_layout.removeWidget(w)
                w.deleteLater()
        self._param_widgets.clear()
        
        self._current_instance_key = None

        effect_id = self._current_effect_id()
        if not effect_id:
            self.plugin_details_label.setText(
                "No effect selected. Choose an effect script from the combo box."
            )
            # Also reset routing UI to defaults
            self._suspend_save = True
            try:
                self._refresh_audio_source_combo()
                self._set_audio_source_in_combo("main")
            finally:
                self._suspend_save = False
            return

        info = self._plugins_by_id.get(effect_id)
        if info is None:
            self.plugin_details_label.setText(
                "Selected effect is not available in the plugin list."
            )
            self._suspend_save = True
            try:
                self._refresh_audio_source_combo()
                self._set_audio_source_in_combo("main")
            finally:
                self._suspend_save = False
            return

        # Details text
        self._update_plugin_details(info)

        # ------------------------------------------------------------------
        # Determine which chain instance we are editing (if any)
        # ------------------------------------------------------------------
        saved_state: Dict[str, Any] = {}
        instance_key: Optional[str] = None

        if (
            self._current_chain_index is not None
            and 0 <= self._current_chain_index < len(self._effect_chain)
        ):
            candidate = self._effect_chain[self._current_chain_index]
            base_id, _ = self._split_chain_entry(candidate)
            if base_id == effect_id:
                instance_key = candidate
                self._current_instance_key = instance_key
                saved_state = self._effect_configs.get(instance_key, {}) or {}

        saved_params = (
            saved_state.get("parameters", {})
            if isinstance(saved_state, dict)
            else {}
        )

        # ------------------------------------------------------------------
        # Build parameter widgets
        # ------------------------------------------------------------------
        params = info.parameters or {}
        for name, param in params.items():
            label = getattr(param, "label", None) or name
            ptype = getattr(param, "type", "float")

            field_widget: QWidget

            if ptype == "bool":
                cb = QCheckBox()
                cb.stateChanged.connect(self._on_params_changed)
                field_widget = cb
                self._param_widgets[name] = cb

            elif ptype == "int":
                # Slider + spinbox pair for integer parameters
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)

                p_min = int(getattr(param, "minimum", getattr(param, "min", 0)))
                p_max = int(getattr(param, "maximum", getattr(param, "max", 100)))
                step = int(getattr(param, "step", 1)) or 1

                sb = QSpinBox(container)
                sb.setMinimum(p_min)
                sb.setMaximum(p_max)
                sb.setSingleStep(step)

                slider = QSlider(Qt.Orientation.Horizontal, container)
                slider.setMinimum(p_min)
                slider.setMaximum(p_max)
                slider.setSingleStep(step)
                slider.setPageStep(max(step, (p_max - p_min) // 10 or 1))

                slider.valueChanged.connect(sb.setValue)
                sb.valueChanged.connect(slider.setValue)
                sb.valueChanged.connect(self._on_params_changed)

                row.addWidget(slider, stretch=3)
                row.addWidget(sb, stretch=1)

                field_widget = container
                self._param_widgets[name] = sb

            elif ptype == "enum":
                cb = QComboBox()
                for choice in getattr(param, "choices", []) or []:
                    cb.addItem(str(choice), choice)
                cb.currentIndexChanged.connect(self._on_params_changed)
                field_widget = cb
                self._param_widgets[name] = cb

            else:  # default: float
                # Slider + double spinbox pair for float parameters
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)

                p_min = float(getattr(param, "minimum", getattr(param, "min", 0.0)))
                p_max = float(getattr(param, "maximum", getattr(param, "max", 1.0)))
                step = float(getattr(param, "step", 0.01)) or 0.01

                if p_max < p_min:
                    p_min, p_max = p_max, p_min

                dsb = QDoubleSpinBox(container)
                dsb.setDecimals(4)
                dsb.setMinimum(p_min)
                dsb.setMaximum(p_max)
                dsb.setSingleStep(step)

                slider = QSlider(Qt.Orientation.Horizontal, container)

                num_steps = max(1, int(round((p_max - p_min) / step)))
                slider.setMinimum(0)
                slider.setMaximum(num_steps)
                slider.setSingleStep(1)
                slider.setPageStep(max(1, num_steps // 10))

                def make_slider_to_spin(
                    spinbox: QDoubleSpinBox,
                    base: float,
                    delta: float,
                ):
                    def _handler(pos: int) -> None:
                        value = base + pos * delta
                        spinbox.blockSignals(True)
                        spinbox.setValue(value)
                        spinbox.blockSignals(False)
                    return _handler

                def make_spin_to_slider(
                    slider_widget: QSlider,
                    base: float,
                    delta: float,
                ):
                    def _handler(value: float) -> None:
                        pos = int(round((value - base) / delta))
                        pos = max(
                            slider_widget.minimum(),
                            min(slider_widget.maximum(), pos),
                        )
                        slider_widget.blockSignals(True)
                        slider_widget.setValue(pos)
                        slider_widget.blockSignals(False)
                    return _handler

                slider.valueChanged.connect(
                    make_slider_to_spin(dsb, p_min, step)
                )
                dsb.valueChanged.connect(
                    make_spin_to_slider(slider, p_min, step)
                )
                dsb.valueChanged.connect(
                    lambda _value: self._on_params_changed()
                )
                slider.valueChanged.connect(
                    lambda _pos: self._on_params_changed()
                )

                row.addWidget(slider, stretch=3)
                row.addWidget(dsb, stretch=1)

                field_widget = container
                self._param_widgets[name] = dsb

            self.params_form_layout.addRow(label + ":", field_widget)

        # ------------------------------------------------------------------
        # Appliquer les valeurs sauvegardées (et l'audio source) SANS sauver
        # ------------------------------------------------------------------
        self._suspend_save = True
        try:
            # 1) Sliders / widgets
            for name, value in saved_params.items():
                w = self._param_widgets.get(name)
                if w is None:
                    continue
                self._set_widget_value(w, value)

            # 2) Routing combo (audio_source_combo)
            self._refresh_audio_source_combo()

            source_id = "main"
            if isinstance(saved_state, dict):
                routing = saved_state.get("routing", {}) or {}
                if isinstance(routing, dict):
                    source_id = routing.get("audio_source", "main")

            self._set_audio_source_in_combo(source_id)
        finally:
            self._suspend_save = False

    def _update_plugin_details(self, info: CoverEffectInfo) -> None:
        text_lines = [
            f"<b>{info.name}</b>",
            f"ID: <code>{info.effect_id}</code>",
            f"Author: {info.author}",
            f"Version: {info.version}",
        ]
        if info.description:
            text_lines.append("")
            text_lines.append(info.description)

        text = "<br>".join(text_lines)
        self.plugin_details_label.setText(text)

    def _current_effect_id(self) -> str:
        effect_id = self.plugin_combo.currentData()
        return effect_id or ""

    # ------------------------------------------------------------------ #
    # Effect chain management (tab 2)                                    #
    # ------------------------------------------------------------------ #

    def _on_add_effect(self) -> None:
        """
        Add a new instance of the currently selected plugin to the chain.

        The instance gets a unique key like "effect_id#N" and an initial
        configuration copied from the current parameter widgets and routing.

        N is always chosen as (max existing index for this effect_id) + 1.
        """
        effect_id = self._current_effect_id()
        if not effect_id:
            QMessageBox.information(
                self,
                "No effect selected",
                "Please choose an effect script from the combo box first.",
            )
            return

        info = self._plugins_by_id.get(effect_id)
        if info is None:
            QMessageBox.warning(
                self,
                "Unknown effect",
                "The selected effect is no longer available.",
            )
            return

        # Find the highest existing index for this effect_id
        max_index_for_effect = 0
        for entry in self._effect_chain:
            base_id, idx = self._split_chain_entry(entry)
            if base_id != effect_id:
                continue
            # Bare effect_id is treated as index 1
            if idx is None:
                idx_value = 1
            else:
                idx_value = idx
            if idx_value > max_index_for_effect:
                max_index_for_effect = idx_value

        instance_index = max_index_for_effect + 1

        # First instance keeps the bare effect_id as key for compatibility
        if instance_index == 1:
            instance_key = effect_id
        else:
            instance_key = f"{effect_id}#{instance_index}"

        # Initial parameters from current UI widgets
        params: Dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            params[name] = self._get_widget_value(widget)
        audio_source = self.audio_source_combo.currentData() or "main"

        self._effect_chain.append(instance_key)
        self._effect_configs[instance_key] = {
            "effect_id": effect_id,
            "parameters": params,
            "routing": {"audio_source": audio_source},
        }

        self._rebuild_chain_list()
        # Select the newly added instance for immediate editing
        new_row = self.chain_list.count() - 1
        if new_row >= 0:
            self.chain_list.setCurrentRow(new_row)
            self._current_chain_index = new_row

        self._save_to_project()

        
    def _on_remove_effect(self) -> None:
        row = self.chain_list.currentRow()
        if row < 0:
            return

        if 0 <= row < len(self._effect_chain):
            instance_key = self._effect_chain[row]
            # Remove from chain
            del self._effect_chain[row]
            # Remove per-instance config
            if instance_key in self._effect_configs:
                del self._effect_configs[instance_key]

        # Adjust current_chain_index
        if self._current_chain_index is not None:
            if self._current_chain_index == row:
                self._current_chain_index = None
            elif self._current_chain_index > row:
                self._current_chain_index -= 1

        self._rebuild_chain_list()
        self._save_to_project()

    def _rebuild_chain_list(self) -> None:
        """
        Rebuild the chain QListWidget from _effect_chain.

        Each entry is displayed as:
            "<Effect name>"          (if single instance)
            "<Effect name> (n)"      (for n-th instance of that effect)
        """
        self.chain_list.clear()

        # First pass: count instances per effect_id
        counts: Dict[str, int] = {}
        for entry in self._effect_chain:
            eff_id, _ = self._split_chain_entry(entry)
            counts[eff_id] = counts.get(eff_id, 0) + 1

        # Second pass: assign running indices per effect_id
        seen: Dict[str, int] = {}

        for entry in self._effect_chain:
            eff_id, explicit_idx = self._split_chain_entry(entry)
            info = self._plugins_by_id.get(eff_id)
            base_name = info.name if info else eff_id

            seen[eff_id] = seen.get(eff_id, 0) + 1
            idx = explicit_idx or seen[eff_id]

            # Only show "(n)" if there is more than one instance
            if counts.get(eff_id, 0) > 1:
                display_name = f"{base_name} ({idx})"
            else:
                display_name = base_name

            item = QListWidgetItem(display_name)
            # Store the full instance key in the item
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.chain_list.addItem(item)

    def _on_chain_selection_changed(self, row: int) -> None:
        """
        When the user selects an entry in the effect chain, reflect that
        choice in the plugin combo and bind the parameter UI to the
        corresponding instance (effect_id or effect_id#N).
        """
        if row < 0 or row >= len(self._effect_chain):
            self._current_chain_index = None
            # No bound instance: rebuild UI in template mode for the current plugin
            self._reload_current_plugin_ui()
            return

        # Instance actually selected in the chain
        self._current_chain_index = row
        instance_key = self._effect_chain[row]
        effect_id, _ = self._split_chain_entry(instance_key)

        # Align the plugin combo with this effect_id
        for i in range(self.plugin_combo.count()):
            if self.plugin_combo.itemData(i) == effect_id:
                self._plugin_combo_change_from_chain = True
                self.plugin_combo.setCurrentIndex(i)
                self._plugin_combo_change_from_chain = False
                break

        # Rebuild UI (sliders + routing combo) for THIS chain entry
        self._reload_current_plugin_ui()

    def _move_selected_row(self, delta: int) -> None:
        row = self.chain_list.currentRow()
        if row < 0:
            return

        new_row = row + delta
        if new_row < 0 or new_row >= self.chain_list.count():
            return

        # Update internal chain
        if 0 <= row < len(self._effect_chain) and 0 <= new_row < len(self._effect_chain):
            self._effect_chain[row], self._effect_chain[new_row] = (
                self._effect_chain[new_row],
                self._effect_chain[row],
            )

        # Update list widget
        item = self.chain_list.takeItem(row)
        self.chain_list.insertItem(new_row, item)
        self.chain_list.setCurrentRow(new_row)
        self._save_to_project()

    def _on_move_up(self) -> None:
        self._move_selected_row(-1)

    def _on_move_down(self) -> None:
        self._move_selected_row(1)

    # ------------------------------------------------------------------ #
    # Audio routing                                                      #
    # ------------------------------------------------------------------ #

    def _refresh_audio_source_combo(self) -> None:
        """
        Populate the audio source combo for the current plugin routing and,
        if available, also the playback source combo for full-chain preview.
        """
        # Build the list of available sources once
        sources: List[tuple[str, str]] = [("Project main audio", "main")]

        if self._project is not None:
            stems_by_model = getattr(self._project, "stems_by_model", {}) or {}
            stem_names = set()
            for _, stems in stems_by_model.items():
                for stem_name in stems.keys():
                    stem_names.add(stem_name)

            for stem_name in sorted(stem_names):
                src_id = f"stem:{stem_name}"
                label = f"{stem_name} stem"
                sources.append((label, src_id))

        # 1) Routing combo for the currently selected plugin
        self.audio_source_combo.blockSignals(True)
        self.audio_source_combo.clear()
        for label, src_id in sources:
            self.audio_source_combo.addItem(label, src_id)
        self.audio_source_combo.blockSignals(False)

        # 2) Playback combo for full-chain preview (if present)
        chain_combo = getattr(self, "chain_audio_source_combo", None)
        if chain_combo is not None:
            # Try to preserve the current selection if possible
            current_id = chain_combo.currentData()
            chain_combo.blockSignals(True)
            chain_combo.clear()
            for label, src_id in sources:
                chain_combo.addItem(label, src_id)

            # Fallback to 'main' when nothing valid is selected
            if current_id is None or not any(src_id == current_id for _, src_id in sources):
                current_id = "main"

            index_to_select = 0
            for i in range(chain_combo.count()):
                if chain_combo.itemData(i) == current_id:
                    index_to_select = i
                    break

            chain_combo.setCurrentIndex(index_to_select)
            chain_combo.blockSignals(False)

    def _set_audio_source_in_combo(self, source_id: str) -> None:
        """
        Select the given source_id in the combo, if present; otherwise fallback to 'main'.
        """
        if not source_id:
            source_id = "main"

        index_to_select = 0
        for i in range(self.audio_source_combo.count()):
            if self.audio_source_combo.itemData(i) == source_id:
                index_to_select = i
                break

        self.audio_source_combo.blockSignals(True)
        self.audio_source_combo.setCurrentIndex(index_to_select)
        self.audio_source_combo.blockSignals(False)

    def _on_audio_source_changed(self, index: int) -> None:
        # Ignore programmatic changes during UI reload.
        if getattr(self, "_suspend_save", False):
            return
        self._save_current_effect_config()

    # ------------------------------------------------------------------ #
    # Parameter handling                                                 #
    # ------------------------------------------------------------------ #

    def _on_params_changed(self, *args: object) -> None:
        # Ignore programmatic changes during UI reload.
        if getattr(self, "_suspend_save", False):
            return
        self._save_current_effect_config()

    def _save_current_effect_config(self) -> None:
        """
        Persist the currently edited instance's parameters + routing into the
        in-memory dict and into the Project.

        The target instance is always the one selected in the effect chain
        (self._current_chain_index), so multiple instances of the same
        effect_id can have independent configurations.
        """
        effect_id = self._current_effect_id()
        if not effect_id:
            return

        if self._current_chain_index is None:
            # No chain entry selected → do not persist (template mode).
            return

        if not (0 <= self._current_chain_index < len(self._effect_chain)):
            return

        instance_key = self._effect_chain[self._current_chain_index]
        base_id, _ = self._split_chain_entry(instance_key)
        if base_id != effect_id:
            # Plugin combo no longer matches this chain entry; do not save.
            return

        params: Dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            params[name] = self._get_widget_value(widget)

        audio_source = self.audio_source_combo.currentData() or "main"

        self._effect_configs[instance_key] = {
            "effect_id": effect_id,
            "parameters": params,
            "routing": {"audio_source": audio_source},
        }

        self._save_to_project()

    def _get_widget_value(self, widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QComboBox):
            # For enums, store the data if present, otherwise text
            data = widget.currentData()
            return data if data is not None else widget.currentText()
        return None

    def _set_widget_value(self, widget: QWidget, value: Any) -> None:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value))
        elif isinstance(widget, QComboBox):
            # Try to match by data first, then text
            idx = -1
            for i in range(widget.count()):
                if widget.itemData(i) == value:
                    idx = i
                    break
            if idx == -1:
                for i in range(widget.count()):
                    if widget.itemText(i) == str(value):
                        idx = i
                        break
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def _migrate_legacy_instance_keys(self) -> None:
        """
        Ensure that in the current in-memory state:

          - at most one chain entry per effect_id uses the plain effect_id key,
          - all additional instances are renamed to 'effect_id#N'
            with their own configuration entry,
          - any pre-existing 'effect_id#N' entries without a config get one
            cloned from the base 'effect_id' config if available.

        This makes sure that each chain entry has its own configuration
        (parameters + routing), so instances of the same effect_id can
        have independent audio sources.
        """
        if not self._effect_chain:
            return

        new_chain: List[str] = []
        counts: Dict[str, int] = {}
        new_configs = dict(self._effect_configs)

        for entry in self._effect_chain:
            effect_id, idx = self._split_chain_entry(entry)

            if idx is None:
                # Plain effect_id: we may need to split into instances
                counts[effect_id] = counts.get(effect_id, 0) + 1
                occurrence = counts[effect_id]

                if occurrence == 1:
                    # First occurrence keeps the bare effect_id
                    key = effect_id
                else:
                    # Subsequent occurrences get suffixed instance keys
                    key = f"{effect_id}#{occurrence}"

                new_chain.append(key)

                # If we have a base config for the plain effect_id, clone it
                # into the new instance key (if not already present).
                if key not in new_configs and effect_id in self._effect_configs:
                    new_configs[key] = copy.deepcopy(self._effect_configs[effect_id])

            else:
                # Already an instance key like "effect_id#N" → keep as is
                new_chain.append(entry)

                # Ensure this instance has its own config; if missing but the
                # base effect_id has a config, clone it.
                if entry not in new_configs and effect_id in self._effect_configs:
                    new_configs[entry] = copy.deepcopy(self._effect_configs[effect_id])

        self._effect_chain = new_chain
        self._effect_configs = new_configs


    # ------------------------------------------------------------------ #
    # Project binding                                                    #
    # ------------------------------------------------------------------ #

    def set_project(self, project: Optional[Project]) -> None:
        """
        Called by MainWindow when the selected project changes.
        """
        self._project = project
        self._current_chain_index = None
        self._current_instance_key = None

        if project is None:
            self.project_label.setText("Current project: (none)")
            self._effect_chain = []
            self._effect_configs = {}
            self._refresh_cover_view()
            self._rebuild_chain_list()
            self._reload_current_plugin_ui()
            return


        self.project_label.setText(f"Current project: {project.name}")

        # Load 2D visual state from project 
        self._effect_configs = dict(
            getattr(project, "cover_visual_effects", {}) or {}
        )
        self._effect_chain = list(
            getattr(project, "cover_visual_chain", []) or []
        )

        # Upgrade old projects where the same effect_id appears multiple times
        # without per-instance keys; this will turn duplicates into
        # effect_id#N entries with their own configs.
        self._migrate_legacy_instance_keys()

        self._refresh_cover_view()
        self._sanitize_chain_against_plugins()
        self._rebuild_chain_list()
        self._reload_current_plugin_ui()



    def _save_to_project(self) -> None:
        """
        Mirror current 2D config back into the Project dataclass and save.
        """
        if self._project is None:
            return

        self._project.cover_visual_effects = dict(self._effect_configs)
        self._project.cover_visual_chain = list(self._effect_chain)
        # Persist changes
        try:
            self._project.save()
        except Exception:
            # Never crash UI because of save issues; silently ignore for now.
            pass

    # ------------------------------------------------------------------ #
    # Cover loading / display                                            #
    # ------------------------------------------------------------------ #

    def _refresh_cover_view(self) -> None:
        """
        Update the cover preview according to the current project.
        """
        self.cover_label.setPixmap(QPixmap())

        if self._project is None:
            self.cover_label.setText("No project selected.")
            return

        if not getattr(self._project, "cover_file", None):
            self.cover_label.setText("This project has no cover file.")
            return

        project_folder = self._project.folder
        cover_path = project_folder / self._project.cover_file

        if not cover_path.exists():
            self.cover_label.setText(f"Cover not found:\n{cover_path.name}")
            return

        pixmap = QPixmap(str(cover_path))
        if pixmap.isNull():
            self.cover_label.setText(f"Failed to load cover:\n{cover_path.name}")
            return

        self._set_pixmap_scaled(pixmap)

    def _set_pixmap_scaled(self, pixmap: QPixmap) -> None:
        target_size = self.cover_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.cover_label.minimumSize()

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cover_label.setPixmap(scaled)
        self.cover_label.setText("")
        
    def _set_pixmap_scaled(self, pixmap: QPixmap) -> None:
        target_size = self.cover_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.cover_label.minimumSize()

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.cover_label.setPixmap(scaled)
        self.cover_label.setText("")

    def _set_chain_preview_pixmap_scaled(self, pixmap: QPixmap) -> None:
        """
        Scale the given pixmap to fit the combined-preview label in the
        'Preview / parameters' tab.
        """
        # Safety: if the label does not exist yet, fall back to main cover.
        label = getattr(self, "chain_preview_label", None)
        if label is None:
            self._set_pixmap_scaled(pixmap)
            return

        target_size = label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = label.minimumSize()

        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setText("")
        

    # ------------------------------------------------------------------ #
    # Preview helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_preview_target_size(self) -> tuple[int, int]:
        """
        Return the target (width, height) used for internal preview frames.

        The values are read from QSettings 'preview/width' and 'preview/height'
        (same keys as the 3D visualizations tab). They are cached on first use.
        """
        if hasattr(self, "_preview_target_size"):
            return self._preview_target_size  # type: ignore[return-value]

        settings = QSettings("Olaf", "OlafApp")
        width = int(settings.value("preview/width", 480))
        height = int(settings.value("preview/height", 270))

        # Clamp to reasonable bounds to avoid absurd values
        width = max(160, min(1920, width))
        height = max(90, min(1080, height))

        self._preview_target_size = (width, height)  # type: ignore[assignment]
        return self._preview_target_size


    def _load_cover_as_array(self) -> Optional[np.ndarray]:
        """
        Load the current project's cover as an RGB uint8 numpy array
        at a *preview* resolution.

        The image is downscaled to the preview size from QSettings
        ('preview/width', 'preview/height') before being converted
        to a numpy array, so that audio-driven previews do not have to
        process huge frames at every tick.
        """
        if self._project is None or not getattr(self._project, "cover_file", None):
            return None

        project_folder = self._project.folder
        cover_path = project_folder / self._project.cover_file

        pixmap = QPixmap(str(cover_path))
        if pixmap.isNull():
            return None

        # Downscale cover for previews (same logic as 3D preview size)
        max_w, max_h = self._get_preview_target_size()
        if max_w > 0 and max_h > 0:
            pixmap = pixmap.scaled(
                max_w,
                max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w = image.width()
        h = image.height()
        ptr = image.bits()
        ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))

        # IMPORTANT: copy the data so we don't depend on QImage's lifetime
        return arr.copy()

    def _array_to_pixmap(self, arr: np.ndarray) -> QPixmap:
        """
        Convert an RGB uint8 array to QPixmap.
        """
        h, w, _ = arr.shape
        image = QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(image.copy())
        
    def _ensure_base_frame(self) -> Optional[np.ndarray]:
        """
        Ensure we have a base RGB frame for previews.

        This prefers a cached frame (from a previous preview) but will
        reload the project cover from disk if needed.
        """
        # Reuse cached base frame if present
        base = getattr(self, "_preview_base_frame", None)
        if base is not None:
            return base

        # Otherwise, load from current project cover
        base = self._load_cover_as_array()
        if base is not None:
            self._preview_base_frame = base
        return base
        

    # ------------------------------------------------------------------ #
    # Live audio-driven preview (single effect)                          #
    # ------------------------------------------------------------------ #

    def _on_preview_single_effect(self) -> None:
        """
        Start a live preview of the currently selected effect, driven by
        the chosen audio source (main track or a stem).

        The parameters are taken directly from the current parameter
        widgets, so this always previews the *visible* configuration,
        i.e. the currently selected instance in the effect chain (if any).
        """
        if self._project is None:
            QMessageBox.information(
                self,
                "No project selected",
                "Please select a project in the Projects tab first.",
            )
            return

        effect_id = self._current_effect_id()
        if not effect_id:
            QMessageBox.information(
                self,
                "No effect selected",
                "Please choose an effect script from the combo box first.",
            )
            return

        # We need librosa for RMS envelope
        if librosa is None:
            QMessageBox.warning(
                self,
                "Missing dependency",
                "The 'librosa' package is required for audio-driven previews.\n"
                "Please install it (e.g. 'pip install librosa').",
            )
            return

        # Resolve the audio path for the currently selected source (main / stem:xxx)
        source_id = self.audio_source_combo.currentData() or "main"
        audio_path = self._resolve_audio_path_for_source(source_id)
        if audio_path is None or not audio_path.exists():
            QMessageBox.warning(
                self,
                "Audio source not available",
                f"Cannot find audio for source '{source_id}'.\n"
                "Make sure the main track and stems are correctly set in the project.",
            )
            return

        # Compute / reuse an RMS envelope for this file.
        fps = 25
        try:
            envelope = self._ensure_audio_envelope(audio_path, fps=fps)
        except Exception as exc:  # defensive
            QMessageBox.critical(
                self,
                "Audio analysis error",
                f"Failed to analyze audio:\n{exc}",
            )
            return

        # Grab the base cover frame as numpy array (RGB).
        base_frame = self._ensure_base_frame()
        if base_frame is None:
            QMessageBox.warning(
                self,
                "No cover",
                "The current project has no valid cover image.",
            )
            return

        # Instantiate the effect with parameters taken from current UI.
        info = self._plugins_by_id.get(effect_id)
        if info is None:
            QMessageBox.warning(
                self,
                "Unknown effect",
                "The selected effect is no longer available in the plugin list.",
            )
            return

        params: Dict[str, Any] = {}
        for name, widget in self._param_widgets.items():
            params[name] = self._get_widget_value(widget)

        try:
            effect = self._manager.create_instance(effect_id, config=params)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Plugin error",
                f"Failed to instantiate effect '{effect_id}':\n{exc}",
            )
            return

        if effect is None:
            QMessageBox.warning(
                self,
                "Plugin error",
                f"Could not create effect instance for '{effect_id}'.",
            )
            return

        # Try to locate the shared QMediaPlayer from the MainWindow parent.
        player = self._get_shared_player()

        # Runtime state for the timer
        self._preview_effect = effect
        self._preview_base_frame = base_frame
        self._preview_envelope = envelope
        self._preview_audio_path = audio_path
        self._preview_fps = fps
        self._preview_original_pixmap = self.cover_label.pixmap()

        # (Re)create the timer if needed.
        if not hasattr(self, "_preview_timer") or self._preview_timer is None:
            self._preview_timer = QTimer(self)
            self._preview_timer.timeout.connect(self._on_preview_tick)

        # Remember a monotonic start time as fallback if no QMediaPlayer.
        self._preview_start_time = time.perf_counter()

        # Start audio playback on the chosen file if we have a shared player.
        if isinstance(player, QMediaPlayer):
            # Ensure previews stop when the global player is paused/stopped
            try:
                player.playbackStateChanged.disconnect(self._on_player_playback_state_changed)
            except Exception:
                # Not previously connected: safe to ignore
                pass
            player.playbackStateChanged.connect(self._on_player_playback_state_changed)

            url = QUrl.fromLocalFile(str(audio_path))
            player.setSource(url)
            player.play()
        else:
            # No player found → visual-only preview, but still time-based.
            player = None

        self._preview_player = player
        interval_ms = int(1000 / max(1, fps))
        self._preview_timer.start(interval_ms)

    def _on_preview_tick(self) -> None:
        """
        Timer callback for live, audio-driven single-effect preview.

        It samples the pre-computed RMS envelope at the current playback
        time and re-applies the selected effect to the cached base frame.
        """
        # Retrieve runtime preview state (defensive: use getattr)
        effect = getattr(self, "_preview_effect", None)
        env = getattr(self, "_preview_envelope", None)
        base_frame = getattr(self, "_preview_base_frame", None)
        player = getattr(self, "_preview_player", None)
        timer = getattr(self, "_preview_timer", None)
        start_time = getattr(self, "_preview_start_time", None)
        fps = getattr(self, "_preview_fps", 25) or 25
        original_pixmap = getattr(self, "_preview_original_pixmap", None)

        # If anything critical is missing, stop preview
        if effect is None or env is None or base_frame is None or timer is None:
            if timer is not None:
                timer.stop()
            return

        # Determine current time t in seconds
        if isinstance(player, QMediaPlayer):
            # QMediaPlayer position is in milliseconds
            t = max(0.0, float(player.position()) / 1000.0)
        else:
            # Fallback to monotonic clock if no player is available
            if start_time is None:
                start_time = time.perf_counter()
                self._preview_start_time = start_time
            t = max(0.0, float(time.perf_counter() - start_time))

        # Stop when reaching the end of the analyzed audio
        if t >= env.duration:
            if isinstance(player, QMediaPlayer):
                player.stop()
            timer.stop()
            # Restore original cover if we have it
            if original_pixmap is not None:
                self._set_pixmap_scaled(original_pixmap)
            return

        # Sample normalized RMS amplitude at time t
        rms = env.rms
        if rms.size == 0:
            amp = 0.0
        else:
            idx = int(t * fps)
            if idx < 0:
                idx = 0
            if idx >= len(rms):
                idx = len(rms) - 1
            amp = float(rms[idx])

        features = FrameFeatures(amp=amp)

        try:
            # Always start from the same base frame to avoid cumulative drift
            frame_in = base_frame
            frame_out = effect.apply_to_frame(frame_in.copy(), t=t, features=features)
        except Exception as exc:
            # Abort preview on plugin error
            if isinstance(player, QMediaPlayer):
                player.stop()
            timer.stop()
            QMessageBox.warning(
                self,
                "Effect error",
                f"Effect raised an exception during live preview:\n{exc}",
            )
            if original_pixmap is not None:
                self._set_pixmap_scaled(original_pixmap)
            return

        if not isinstance(frame_out, np.ndarray) or frame_out.ndim != 3 or frame_out.shape[2] != 3:
            # Invalid frame, just ignore this tick
            return

        frame_out = np.clip(frame_out, 0, 255).astype(np.uint8)
        pixmap = self._array_to_pixmap(frame_out)
        self._set_pixmap_scaled(pixmap)


    def _on_preview_chain(self) -> None:
        """
        Apply the full effect chain once to the cover, using a boosted
        dummy amplitude, and show the result in the cover preview.

        This is a *static* preview:
          - we do not play audio,
          - we assume a single frame with amp=1.0,
          - each effect is instantiated with the parameters stored in
            self._effect_configs[effect_id]["parameters"].

        The chain is applied in order, starting from the original cover.
        """
        if self._project is None:
            QMessageBox.information(
                self,
                "No project",
                "Select a project with a cover first.",
            )
            return

        if not self._effect_chain:
            QMessageBox.information(
                self,
                "Empty chain",
                "The effect chain is empty. Add some effects first.",
            )
            return

        base_img = self._load_cover_as_array()
        if base_img is None:
            QMessageBox.warning(
                self,
                "No cover",
                "Could not load the project cover image.",
            )
            return

        # Start from the original cover
        img = base_img
        # Max amplitude for preview so that effects are clearly visible
        features = FrameFeatures(amp=1.0)

        # Apply each effect in order
        for entry in self._effect_chain:
            effect_id, _ = self._split_chain_entry(entry)
            info = self._plugins_by_id.get(effect_id)
            if info is None:
                # Plugin no longer available; skip but keep the chain going
                continue

            cfg_entry = self._effect_configs.get(entry)
            if cfg_entry is None:
                # Fallback to legacy per-effect config
                cfg_entry = self._effect_configs.get(effect_id, {})

            base_params = (
                cfg_entry.get("parameters", {})
                if isinstance(cfg_entry, dict)
                else {}
            )
            params = dict(base_params)

            # If the effect exposes a 'probability' parameter, force it to 1.0
            # during this preview so stochastic effects are fully visible.
            try:
                if "probability" in (info.parameters or {}):
                    params["probability"] = 1.0
            except Exception:
                # Defensive: never break the preview because of metadata issues
                pass

            effect_instance = self._manager.create_instance(effect_id, config=params)
            if effect_instance is None:
                # Broken plugin; skip it
                continue

            try:
                # Minimal sequence context for potential precomputation
                effect_instance.on_sequence_start(duration=1.0, fps=1)
                img = effect_instance.apply_to_frame(
                    img,
                    t=0.0,
                    features=features,
                )
            except Exception:
                # Never crash the UI because of a single effect: skip and continue
                continue

            if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
                # Effect returned something invalid; restore last valid frame
                img = base_img
                break


        img = np.clip(img, 0, 255).astype(np.uint8)
        pixmap = self._array_to_pixmap(img)
        # Show the result directly on the main cover preview
        self._set_pixmap_scaled(pixmap)


    def _on_preview_chain_live(self) -> None:
        """
        Start a live, audio-driven preview of the full effect chain.

        Each effect in the chain uses its OWN routed audio source
        (main or a given stem) to compute its amplitude, while the
        audio actually played back to the user can be chosen separately
        via 'chain_audio_source_combo'.
        """
        if self._project is None:
            QMessageBox.information(
                self,
                "No project",
                "Select a project with a cover first.",
            )
            return

        if not self._effect_chain:
            QMessageBox.information(
                self,
                "Empty chain",
                "The effect chain is empty. Add some effects first.",
            )
            return

        # Choose playback source (what the user hears)
        playback_source_id = "main"
        if hasattr(self, "chain_audio_source_combo"):
            playback_source_id = self.chain_audio_source_combo.currentData() or "main"

        playback_audio_path = self._resolve_audio_path_for_source(str(playback_source_id))
        if playback_audio_path is None or not playback_audio_path.exists():
            QMessageBox.warning(
                self,
                "Audio not found",
                "Could not resolve the playback audio file for the selected source.",
            )
            return

        # Fixed FPS, same as single-effect preview
        fps = 25
        try:
            playback_env = self._ensure_audio_envelope(playback_audio_path, fps=fps)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Audio analysis error",
                f"Failed to analyze playback audio for chain preview:\n{exc}",
            )
            return

        base_frame = self._ensure_base_frame()
        if base_frame is None:
            QMessageBox.warning(
                self,
                "No cover",
                "The current project has no valid cover image.",
            )
            return

        # Instantiate all effects in the chain with their current parameters,
        # and collect their individual audio routing.
        chain_instances: List[Any] = []
        chain_sources: List[str] = []
        routed_source_ids: set[str] = set()

        for entry in self._effect_chain:
            effect_id, _ = self._split_chain_entry(entry)
            info = self._plugins_by_id.get(effect_id)
            if info is None:
                continue

            config_entry = self._effect_configs.get(entry)
            if config_entry is None:
                # Fallback to legacy per-effect config
                config_entry = self._effect_configs.get(effect_id, {})

            base_params = (
                config_entry.get("parameters", {})
                if isinstance(config_entry, dict)
                else {}
            )
            params = dict(base_params)

            routing = (
                config_entry.get("routing", {})
                if isinstance(config_entry, dict)
                else {}
            )
            source_id = routing.get("audio_source", "main") or "main"

            try:
                effect_instance = self._manager.create_instance(
                    effect_id,
                    config=params,
                )
            except Exception:
                # Skip effects that cannot be instantiated
                continue

            if effect_instance is None:
                continue

            try:
                effect_instance.on_sequence_start(
                    duration=playback_env.duration,
                    fps=fps,
                )
            except Exception:
                # Non-fatal: the effect can still run without this hook
                pass

            chain_instances.append(effect_instance)
            chain_sources.append(source_id)
            routed_source_ids.add(source_id)


        if not chain_instances:
            QMessageBox.warning(
                self,
                "Chain error",
                "Could not instantiate any effect in the chain for preview.",
            )
            return

        # Build / reuse envelopes for each routed source used by the chain
        per_source_env: Dict[str, Optional[AudioEnvelope]] = {}

        for source_id in routed_source_ids:
            audio_path = self._resolve_audio_path_for_source(source_id)
            if audio_path is None or not audio_path.exists():
                per_source_env[source_id] = None
                continue

            try:
                per_source_env[source_id] = self._ensure_audio_envelope(
                    audio_path,
                    fps=fps,
                )
            except Exception:
                per_source_env[source_id] = None

        # Ensure playback source envelope is present in the map as fallback
        per_source_env.setdefault(playback_source_id, playback_env)

        # Try to locate the shared QMediaPlayer from the MainWindow parent
        player = self._get_shared_player()

        # Stop any existing single-effect preview
        preview_timer = getattr(self, "_preview_timer", None)
        if preview_timer is not None:
            preview_timer.stop()
        preview_player = getattr(self, "_preview_player", None)
        if isinstance(preview_player, QMediaPlayer):
            preview_player.stop()

        # Stop any previous chain preview timer
        chain_timer = getattr(self, "_chain_preview_timer", None)
        if chain_timer is not None:
            chain_timer.stop()
        chain_player = getattr(self, "_chain_preview_player", None)
        if isinstance(chain_player, QMediaPlayer):
            chain_player.stop()

        # Store runtime state for the chain preview
        self._chain_preview_effects = chain_instances
        self._chain_preview_sources = chain_sources
        self._chain_preview_envelopes = per_source_env
        self._chain_preview_playback_env = playback_env
        self._chain_preview_base_frame = base_frame
        self._chain_preview_audio_path = playback_audio_path
        self._chain_preview_fps = fps
        self._chain_preview_original_pixmap = self.cover_label.pixmap()

        # (Re)create the chain preview timer if needed
        if not hasattr(self, "_chain_preview_timer") or self._chain_preview_timer is None:
            self._chain_preview_timer = QTimer(self)
            self._chain_preview_timer.timeout.connect(self._on_chain_preview_tick)

        # Monotonic start time fallback if there is no QMediaPlayer
        self._chain_preview_start_time = time.perf_counter()

        # Start audio playback on the chosen playback file if we have a shared player
        if isinstance(player, QMediaPlayer):
            # Ensure previews stop when the global player is paused/stopped
            try:
                player.playbackStateChanged.disconnect(self._on_player_playback_state_changed)
            except Exception:
                # Not previously connected: safe to ignore
                pass
            player.playbackStateChanged.connect(self._on_player_playback_state_changed)

            url = QUrl.fromLocalFile(str(playback_audio_path))
            player.setSource(url)
            player.play()
        else:
            player = None  # visual-only preview, but still time-based

        self._chain_preview_player = player
        interval_ms = int(1000 / max(1, fps))
        self._chain_preview_timer.start(interval_ms)


    def _on_chain_preview_tick(self) -> None:
        """
        Timer callback for live, audio-driven *chain* preview.

        It samples the pre-computed RMS envelopes at the current playback
        time and reapplies the whole effect chain on the cached base frame.

        Each effect uses the envelope corresponding to its own routed
        audio source (main track or a selected stem). The preview stops
        when the playback envelope reaches its end.
        """
        effects = getattr(self, "_chain_preview_effects", None)
        sources = getattr(self, "_chain_preview_sources", None)
        env_map = getattr(self, "_chain_preview_envelopes", None)
        playback_env = getattr(self, "_chain_preview_playback_env", None)
        base_frame = getattr(self, "_chain_preview_base_frame", None)
        player = getattr(self, "_chain_preview_player", None)
        timer = getattr(self, "_chain_preview_timer", None)
        start_time = getattr(self, "_chain_preview_start_time", None)
        original_pixmap = getattr(self, "_chain_preview_original_pixmap", None)

        if (
            not effects
            or sources is None
            or env_map is None
            or playback_env is None
            or base_frame is None
            or timer is None
        ):
            if timer is not None:
                timer.stop()
            return

        fps = int(getattr(self, "_chain_preview_fps", 25) or 25)

        # Compute current time t
        if isinstance(player, QMediaPlayer):
            t = max(0.0, float(player.position()) / 1000.0)
        else:
            if start_time is None:
                start_time = time.perf_counter()
                self._chain_preview_start_time = start_time
            t = max(0.0, float(time.perf_counter() - start_time))

        # Stop when reaching end of playback audio
        if t >= playback_env.duration:
            if isinstance(player, QMediaPlayer):
                player.stop()
            timer.stop()
            if original_pixmap is not None:
                self._set_pixmap_scaled(original_pixmap)
            return

        try:
            frame = base_frame
            # Apply each effect with its own routed envelope
            for effect, source_id in zip(effects, sources):
                env = env_map.get(source_id)
                amp = 0.0

                if env is not None and env.rms.size > 0:
                    # Use this source's fps; normally equals 'fps'
                    eff_fps = int(getattr(env, "fps", fps) or fps)
                    idx = int(t * eff_fps)
                    if idx < 0:
                        idx = 0
                    if idx >= len(env.rms):
                        idx = len(env.rms) - 1
                    amp = float(env.rms[idx])

                features = FrameFeatures(amp=amp)
                frame = effect.apply_to_frame(frame.copy(), t=t, features=features)

        except Exception as exc:
            if isinstance(player, QMediaPlayer):
                player.stop()
            timer.stop()
            QMessageBox.warning(
                self,
                "Effect chain error",
                f"Effect chain raised an exception during live preview:\n{exc}",
            )
            if original_pixmap is not None:
                self._set_pixmap_scaled(original_pixmap)
            return

        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            # Invalid frame, ignore this tick
            return

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        pixmap = self._array_to_pixmap(frame)
        self._set_pixmap_scaled(pixmap)


    def _on_player_playback_state_changed(
        self,
        state: QMediaPlayer.PlaybackState,
    ) -> None:
        """
        Stop any active visualization preview timers when the shared audio
        player is paused or stopped via the global transport controls.
        """
        # We only care about transitions away from PlayingState
        if state == QMediaPlayer.PlaybackState.PlayingState:
            return

        # Stop single-effect preview timer if active
        preview_timer = getattr(self, "_preview_timer", None)
        if preview_timer is not None and preview_timer.isActive():
            preview_timer.stop()

        # Stop chain live preview timer if active
        chain_timer = getattr(self, "_chain_preview_timer", None)
        if chain_timer is not None and chain_timer.isActive():
            chain_timer.stop()


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

        # Stem audio (first matching stem across models)
        if source_id.startswith("stem:"):
            stem_name = source_id.split(":", 1)[1]
            stems_by_model = getattr(self._project, "stems_by_model", {}) or {}
            for _, stems in stems_by_model.items():
                rel_path = stems.get(stem_name)
                if rel_path:
                    return self._project.folder / rel_path

        return None
        
    def _ensure_audio_envelope(self, audio_path: Path, fps: int) -> AudioEnvelope:
        """
        Compute (or reuse from cache) a normalized RMS envelope for a given
        audio file at a given frame rate (fps).
        """
        if not hasattr(self, "_audio_cache"):
            self._audio_cache: Dict[tuple, AudioEnvelope] = {}

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
 
    def _get_shared_player(self) -> Optional[QMediaPlayer]:
        """
        Walk up the parent chain to find the MainWindow that holds the
        shared QMediaPlayer ('player' attribute).
        """
        widget = self.parent()
        while widget is not None and not hasattr(widget, "player"):
            widget = widget.parent()

        player = getattr(widget, "player", None) if widget is not None else None
        if isinstance(player, QMediaPlayer):
            return player
        return None


