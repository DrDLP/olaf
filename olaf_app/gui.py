from __future__ import annotations

import sys
from typing import Optional
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFontDatabase, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QScrollArea,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
)


from .project_manager import Project
from .projects_tab import ProjectsTab
from .stems_tab import StemsTab
from .settings_tab import SettingsTab
from .vocal_tab import VocalTab
from .visualizations_tab import VisualizationsTab
from .cover_visualizations_tab import CoverVisualizationsTab
from .lyrics_visualizations_tab import LyricsVisualizationsTab
from .export_tab import ExportTab

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Position de drag pour la barre de titre custom
        self._drag_pos = None

        self.setWindowTitle("Olaf")

        # Safe loading of the custom application icon
        icon_path = Path(__file__).resolve().parent.parent / "OLAF.ico"
        if icon_path.is_file():
            self._app_icon = QIcon(str(icon_path))
            self.setWindowIcon(self._app_icon)
        else:
            self._app_icon = QIcon()

        self.resize(1200, 800)

        # Use a frameless window so we can draw our own title bar
        # self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)

        # ------------------------------------------------------------------
        # 1) Shared media player (audio for all tabs)
        # ------------------------------------------------------------------
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self._current_project: Optional[Project] = None

        # Global audio volume (0–100%), persisted in QSettings
        self._volume_percent: int = 100
        self._init_audio_volume()


        # ------------------------------------------------------------------
        # 2) Central widget = custom title bar + tabs + bottom player bar
        # ------------------------------------------------------------------
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 8)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        # ------------------------------------------------------------------
        # 3) Top-level tabs (wrapped in scroll areas)
        # ------------------------------------------------------------------
        self.tabs = QTabWidget(central)

        self.projects_tab = ProjectsTab(self.player, parent=self)
        self.stems_tab = StemsTab(self.player, parent=self)
        self.vocal_tab = VocalTab(self.player, parent=self)
        self.visualizations_3d_tab = VisualizationsTab(self.player, parent=self)
        self.cover_visualizations_tab = CoverVisualizationsTab(parent=self)
        self.lyrics_visualizations_tab = LyricsVisualizationsTab(self.player, parent=self)
        self.settings_tab = SettingsTab(parent=self)
        self.export_tab = ExportTab(parent=self)

        def _make_scroll_container(inner: QWidget) -> QScrollArea:
            container = QScrollArea(self.tabs)
            container.setWidgetResizable(True)
            container.setWidget(inner)
            container.setFrameShape(QScrollArea.Shape.NoFrame)
            return container

        self.tabs.addTab(_make_scroll_container(self.projects_tab), "Projects")
        self.tabs.addTab(_make_scroll_container(self.stems_tab), "Stems")
        self.tabs.addTab(_make_scroll_container(self.vocal_tab), "Vocal")
        self.tabs.addTab(_make_scroll_container(self.visualizations_3d_tab), "3D visualizations")
        self.tabs.addTab(_make_scroll_container(self.cover_visualizations_tab), "2D visualizations")
        self.tabs.addTab(_make_scroll_container(self.lyrics_visualizations_tab), "Lyrics visuals")
        self.tabs.addTab(_make_scroll_container(self.settings_tab), "Settings")
        self.tabs.addTab(_make_scroll_container(self.export_tab), "Export")

        # Initialize the global UI theme from QSettings and track changes
        self._init_theme()

        # Tabs fill all the space above the global player bar
        central_layout.addWidget(self.tabs)

        # Global bottom player bar (always visible)
        self._build_global_player_bar(central_layout)


        # Listen to theme changes coming from the Settings tab
        if hasattr(self.settings_tab, "themeChanged"):
            self.settings_tab.themeChanged.connect(self._apply_theme)
        if hasattr(self.settings_tab, "scaleChanged"):
            self.settings_tab.scaleChanged.connect(self._apply_scale)
        if hasattr(self.settings_tab, "volumeChanged"):
            self.settings_tab.volumeChanged.connect(self.set_global_volume)

        # Wiring between tabs and player
        self.projects_tab.projectSelected.connect(self.on_project_selected)

        self.player.positionChanged.connect(self.projects_tab.on_position_changed)
        self.player.durationChanged.connect(self.projects_tab.on_duration_changed)
        self.player.mediaStatusChanged.connect(self.projects_tab.on_media_status_changed)

        self.player.positionChanged.connect(self.stems_tab.on_position_changed)
        self.player.durationChanged.connect(self.stems_tab.on_duration_changed)

        self.player.positionChanged.connect(self.vocal_tab.on_position_changed)

        self.player.positionChanged.connect(
            self.lyrics_visualizations_tab.on_position_changed
        )
        self.player.durationChanged.connect(
            self.lyrics_visualizations_tab.on_duration_changed
        )

        # Initial project list load
        self.projects_tab.refresh_projects()


    def _init_audio_volume(self) -> None:
        """
        Load the global audio volume from QSettings and apply it
        to the shared QAudioOutput instance.

        The value is stored as an integer percentage (0–100).
        """
        settings = QSettings("Olaf", "OlafApp")
        stored = settings.value("audio/volume_percent", 100, type=int)
        try:
            vol = int(stored)
        except Exception:
            vol = 100

        vol = max(0, min(100, vol))
        self._volume_percent = vol
        # QAudioOutput expects a float between 0.0 and 1.0
        self.audio_output.setVolume(vol / 100.0)

    def set_global_volume(self, volume_percent: int) -> None:
        """
        Update the global audio volume (0–100 %) from any UI control.

        This method:
          - clamps the value,
          - applies it to QAudioOutput,
          - persists it in QSettings,
          - synchronizes the bottom-bar slider and the Settings tab slider.
        """
        try:
            vol = int(volume_percent)
        except Exception:
            vol = 100

        vol = max(0, min(100, vol))
        self._volume_percent = vol
        self.audio_output.setVolume(vol / 100.0)

        # Persist in QSettings
        settings = QSettings("Olaf", "OlafApp")
        settings.setValue("audio/volume_percent", vol)

        # Sync bottom global slider if it exists
        if hasattr(self, "global_volume_slider"):
            self.global_volume_slider.blockSignals(True)
            self.global_volume_slider.setValue(vol)
            self.global_volume_slider.blockSignals(False)

        # Sync Settings tab slider + label if they exist
        if hasattr(self, "settings_tab") and hasattr(self.settings_tab, "volume_slider"):
            self.settings_tab.volume_slider.blockSignals(True)
            self.settings_tab.volume_slider.setValue(vol)
            self.settings_tab.volume_slider.blockSignals(False)
            if hasattr(self.settings_tab, "lbl_volume_value"):
                self.settings_tab.lbl_volume_value.setText(f"{vol} %")



    # ----------------------------------------------------------------------
    # Slots
    # ----------------------------------------------------------------------

    def on_project_selected(self, project: Optional[Project]):
        """Called when ProjectsTab selection changes."""
        self._current_project = project
        self.stems_tab.set_project(project)
        self.vocal_tab.set_project(project)
        self.visualizations_3d_tab.set_project(project)
        self.cover_visualizations_tab.set_project(project)
        self.lyrics_visualizations_tab.set_project(project)
        self.export_tab.set_project(project)
        # Refresh the global project banner whenever selection changes
        self.update_now_playing("")

    def _build_title_bar(self, parent_layout: QVBoxLayout) -> None:
        """
        Create a custom title bar that matches the app theme.

        It shows:
          - Olaf icon + title,
          - window control buttons (minimize / maximize / close),
          - drag support when clicking and dragging the bar.
        """
        bar = QWidget(self)
        bar.setObjectName("TitleBar")

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        # Small app icon on the left
        icon_label = QLabel(bar)
        if not self._app_icon.isNull():
            icon_label.setPixmap(self._app_icon.pixmap(16, 16))
        layout.addWidget(icon_label)

        # App name
        title_label = QLabel("Olaf", bar)
        layout.addWidget(title_label)

        layout.addStretch(1)

        # Window control buttons
        self._btn_minimize = QPushButton("–", bar)
        self._btn_minimize.setObjectName("TitleButton")
        self._btn_minimize.clicked.connect(self.showMinimized)

        self._btn_maximize = QPushButton("▢", bar)
        self._btn_maximize.setObjectName("TitleButton")
        self._btn_maximize.clicked.connect(self._toggle_maximize_restore)

        self._btn_close = QPushButton("✕", bar)
        self._btn_close.setObjectName("TitleButtonClose")
        self._btn_close.clicked.connect(self.close)

        layout.addWidget(self._btn_minimize)
        layout.addWidget(self._btn_maximize)
        layout.addWidget(self._btn_close)

        # Enable dragging the window by the title bar
        bar.mousePressEvent = self._on_title_bar_mouse_press  # type: ignore[assignment]
        bar.mouseMoveEvent = self._on_title_bar_mouse_move    # type: ignore[assignment]
        bar.mouseDoubleClickEvent = self._on_title_bar_double_click  # type: ignore[assignment]

        parent_layout.addWidget(bar)


    def _build_global_player_bar(self, parent_layout: QVBoxLayout) -> None:
        """Create the global bottom player bar and wire it to the shared QMediaPlayer."""
        bar = QWidget(self)
        bar.setObjectName("GlobalPlayerBar")  
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(8, 4, 8, 4)
        bar_layout.setSpacing(8)



        self.now_playing_label = QLabel("No project selected", bar)
        self.now_playing_label.setMinimumWidth(320)
        self.now_playing_label.setStyleSheet(
            "font-size: 13pt; font-weight: bold; padding: 4px 8px;"
        )


        self.global_play_pause_btn = QPushButton("Play", bar)
        self.global_stop_btn = QPushButton("Stop", bar)

        self.global_position_slider = QSlider(Qt.Orientation.Horizontal, bar)
        self.global_position_slider.setRange(0, 0)

        # Neon-style track slider: dark blue = already played, light blue = remaining
        self.global_position_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 18px;
                margin: 0px;
                border-radius: 9px;
                background: #050a14;
            }
            QSlider::handle:horizontal {
                background: #0b1020;
                border: 1px solid #66C2FF;
                width: 10px;
                margin: -6px 0;
                border-radius: 5px;
            }
            /* left side (already played) = darker blue */
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0.0 #004F9E,
                    stop: 1.0 #00284F
                );
                border-radius: 9px;
            }
            /* right side (not yet played) = lighter blue */
            QSlider::add-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0.0 #66C2FF,
                    stop: 1.0 #99D6FF
                );
                border-radius: 9px;
            }
            """
        )

        # Small global volume slider (0–100 %), synced with Settings
        self.global_volume_slider = QSlider(Qt.Orientation.Horizontal, bar)
        self.global_volume_slider.setRange(0, 100)
        self.global_volume_slider.setFixedWidth(90)
        self.global_volume_slider.setValue(self._volume_percent)
        self.global_volume_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 8px;
                margin: 0px;
                border-radius: 4px;
                background: #1a1a2a;
            }
            QSlider::handle:horizontal {
                background: #66C2FF;
                border: 1px solid #004F9E;
                width: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: #004F9E;
                border-radius: 4px;
            }
            QSlider::add-page:horizontal {
                background: #404060;
                border-radius: 4px;
            }
            """
        )

        bar_layout.addWidget(self.now_playing_label, 1)
        bar_layout.addWidget(self.global_play_pause_btn)
        bar_layout.addWidget(self.global_stop_btn)
        bar_layout.addWidget(self.global_position_slider, 2)
        bar_layout.addWidget(self.global_volume_slider)


        parent_layout.addWidget(bar)

        # Button actions
        self.global_play_pause_btn.clicked.connect(self._on_global_play_pause_clicked)
        self.global_stop_btn.clicked.connect(self.player.stop)
        self.global_position_slider.sliderMoved.connect(self._on_global_slider_moved)
        self.global_volume_slider.valueChanged.connect(self.set_global_volume)

        # Sync with player
        self.player.playbackStateChanged.connect(self._on_player_state_changed_global)
        self.player.positionChanged.connect(self._on_player_position_changed_global)
        self.player.durationChanged.connect(self._on_player_duration_changed_global)

        # Connect tab signals -> update "now playing"
        if hasattr(self.projects_tab, "audioStarted"):
            self.projects_tab.audioStarted.connect(self.update_now_playing)
        if hasattr(self.stems_tab, "audioStarted"):
            self.stems_tab.audioStarted.connect(self.update_now_playing)
        if hasattr(self.vocal_tab, "audioStarted"):
            self.vocal_tab.audioStarted.connect(self.update_now_playing)

    # ------------------------------------------------------------------
    # Global player bar helpers
    # ------------------------------------------------------------------

    def _toggle_maximize_restore(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # -------------------- Title bar mouse handling -------------------- #

    def _on_title_bar_mouse_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def _on_title_bar_mouse_move(self, event) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_pos is not None:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def _on_title_bar_double_click(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize_restore()
            event.accept()


    # ------------------------------------------------------------------
    # Global UI theme management
    # ------------------------------------------------------------------

    def _init_theme(self) -> None:
        """Load theme and UI scale from QSettings and apply stylesheet.

        Defaults to the neon (dark blue) theme at 100% scale.
        """
        settings = QSettings("Olaf", "OlafApp")
        # Theme key is a short string: "dark", "light", "neon", ...
        self._theme_key = settings.value("ui/theme", "neon", type=str)
        # UI scale is stored as a percentage (e.g. 75, 100, 125).
        try:
            self._ui_scale_percent = int(settings.value("ui/scale_percent", 100))
        except Exception:
            self._ui_scale_percent = 100

        self._apply_theme(self._theme_key)


    def _apply_theme(self, theme_key: str) -> None:
        """
        Apply a global Qt stylesheet based on the given theme key.

        The stylesheet is applied at the QApplication level so that
        all windows and widgets share the same appearance.
        """
        theme_key = (theme_key or "neon").lower()
        self._theme_key = theme_key


        # Dark theme: compact, neutral background suited for media work.
        dark_css = """
            QWidget {
                background-color: #191919;
                color: #EEEEEE;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                font-size: 10pt;
            }
            
            #MainCentralWidget {
                border: 1px solid #FFFFFF;
            }

            #GlobalPlayerBar {
                border: 0px solid #FFFFFF;
                border-radius: 0px;
                background-color: #151515;
            }            
            
            QMainWindow::separator {
                background: #303030;
            }

            QTabWidget::pane {
                border: 1px solid #333333;
                background: #202020;
            }

            QTabBar::tab {
                background: #282828;
                border: 1px solid #3A3A3A;
                padding: 6px 12px;
                margin-right: 2px;
                font-weight: bold;
            }


            QTabBar::tab:selected {
                background: #383838;
                border-bottom: 2px solid #46C2FF;
            }

            QTabBar::tab:hover {
                background: #303030;
            }

            QGroupBox {
                border: 1px solid #3A3A3A;
                margin-top: 12px;
                padding: 12px 12px 16px 12;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }

            QLabel {
                background-color: transparent;
            }

            /* ----------------- EMPHASISED CONTROLS (DARK) ----------------- */

            QPushButton {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #3A3A3A,
                    stop: 1.0 #242424
                );
                border: 1px solid #5A5A5A;
                border-radius: 4px;
                padding: 6px 14px;
                color: #F5F5F5;
                font-weight: bold;
                /* subtle pseudo-relief: lighter top, darker bottom */
                border-top-color: #767676;
                border-bottom-color: #181818;
            }


            QPushButton:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #484848,
                    stop: 1.0 #2E2E2E
                );
            }

            QPushButton:pressed {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #1F1F1F,
                    stop: 1.0 #141414
                );
                border-top-color: #101010;
                border-bottom-color: #505050;
            }

            QPushButton:disabled {
                background-color: #202020;
                color: #777777;
                border: 1px solid #333333;
            }

            QComboBox {
                background-color: #222222;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 24px 2px 6px;
                font-weight: bold;
            }


            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }

            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #CCCCCC;
                margin-right: 4px;
            }

            QComboBox:on {
                background-color: #2A2A2A;
            }

            QCheckBox {
                spacing: 6px;
                font-weight: bold;
            }

            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #666666;
                background-color: #262626;
            }

            QCheckBox::indicator:checked {
                border: 1px solid #46C2FF;
                background-color: #46C2FF;
            }

            QCheckBox::indicator:checked:disabled {
                border: 1px solid #888888;
                background-color: #555555;
            }

            QLineEdit,
            QSpinBox,
            QComboBox QAbstractItemView,
            QPlainTextEdit,
            QTextEdit {
                background-color: #222222;
                border: 1px solid #444444;
                border-radius: 3px;
                padding: 2px 4px;
                selection-background-color: #46C2FF;
                selection-color: #000000;
            }

            QListWidget {
                background-color: #202020;
                border: 1px solid #3A3A3A;
            }

            QListWidget::item:selected {
                background: #35597A;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #303030;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #46C2FF;
                border: 1px solid #1A4B70;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }

            QScrollBar:vertical {
                background: #202020;
                width: 12px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: #444444;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """

        # Light theme: more neutral, high contrast for bright environments.
        light_css = """
            QWidget {
                background-color: #F4F4F4;
                color: #202020;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                font-size: 10pt;
            }
            #GlobalPlayerBar {
                border: 2px solid #CCCCCC;
                border-radius: 8px;
                background-color: #F8F8F8;
            }

            QTabWidget::pane {
                border: 1px solid #C0C0C0;
                background: #FFFFFF;
            }

            QTabBar::tab {
                background: #E8E8E8;
                border: 1px solid #C0C0C0;
                padding: 6px 12px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background: #FFFFFF;
                border-bottom: 2px solid #007ACC;
            }

            QTabBar::tab:hover {
                background: #F0F0F0;
            }

            QGroupBox {
                border: 1px solid #D0D0D0;
                margin-top: 8px;
                padding-top: 16px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }

            /* ----------------- EMPHASISED CONTROLS (LIGHT) ----------------- */

            QPushButton {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #FFFFFF,
                    stop: 1.0 #E4E4E4
                );
                border: 1px solid #B8B8B8;
                border-radius: 4px;
                padding: 5px 12px;
                color: #202020;
                border-top-color: #FFFFFF;
                border-bottom-color: #A0A0A0;
            }

            QPushButton:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #FFFFFF,
                    stop: 1.0 #D8D8D8
                );
            }

            QPushButton:pressed {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #D8D8D8,
                    stop: 1.0 #C2C2C2
                );
                border-top-color: #A0A0A0;
                border-bottom-color: #FFFFFF;
            }

            QPushButton:disabled {
                background-color: #F0F0F0;
                color: #A0A0A0;
                border: 1px solid #D0D0D0;
            }

            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #B8B8B8;
                border-radius: 3px;
                padding: 2px 24px 2px 6px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #B8B8B8;
            }

            QComboBox::down-arrow {
                image: none;
                width: 0;
                height: 0;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 7px solid #404040;
                margin-right: 4px;
            }

            QComboBox:on {
                background-color: #F4F4F4;
            }

            QCheckBox {
                spacing: 6px;
            }

            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 1px solid #B0B0B0;
                background-color: #FFFFFF;
            }

            QCheckBox::indicator:checked {
                border: 1px solid #007ACC;
                background-color: #007ACC;
            }

            QCheckBox::indicator:checked:disabled {
                border: 1px solid #C0C0C0;
                background-color: #B0B0B0;
            }

            QLineEdit,
            QSpinBox,
            QComboBox QAbstractItemView,
            QPlainTextEdit,
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #C0C0C0;
                border-radius: 3px;
                padding: 2px 4px;
                selection-background-color: #007ACC;
                selection-color: #FFFFFF;
            }

            QListWidget {
                background-color: #FFFFFF;
                border: 1px solid #C0C0C0;
            }

            QListWidget::item:selected {
                background: #CCE4FF;
            }

            QSlider::groove:horizontal {
                height: 6px;
                background: #D0D0D0;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #007ACC;
                border: 1px solid #005A9E;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }

            QScrollBar:vertical {
                background: #F0F0F0;
                width: 12px;
                margin: 0px;
            }

            QScrollBar::handle:vertical {
                background: #C4C4C4;
                border-radius: 6px;
                min-height: 20px;
            }

            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """

        # Neon studio theme: a bit more playful, tuned for audio/visual work.
        neon_css = """
            QWidget {
                background-color: #050510;
                color: #E8E8FF;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
                font-size: 10pt;
            }

            /* Main tab frame */
            QTabWidget::pane {
                border: 1px solid #3A114A;
                background: #090918;
            }
            #GlobalPlayerBar {
                border: 2px solid #FFFFFF;
                border-radius: 8px;
                background-color: #050510;
            }

            /* Top tab bar */
            QTabBar::tab {
                background: #101027;
                border: 1px solid #2A1D45;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 8px 18px;
                margin: 0px 2px 0px 0px;
                color: #C8C8FF;
            }

            QTabBar::tab:selected {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0.0 #251842,
                    stop: 1.0 #3A114A
                );
                color: #FFFFFF;
                border-color: #FF3CF0;
                border-bottom: 2px solid #090918; /* visually merges with pane */
            }

            QTabBar::tab:hover {
                background: #18152C;
            }

            /* Group boxes */
            QGroupBox {
                border: 1px solid #2A1D45;
                margin-top: 8px;
                padding-top: 16px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #FF7BFF;
            }

            /* Custom title bar */
            #TitleBar {
                background-color: #050510;
                border-bottom: 1px solid #3A114A;
            }

            #TitleBar QLabel {
                font-weight: 600;
            }

            #TitleBar QPushButton#TitleButton,
            #TitleBar QPushButton#TitleButtonClose {
                min-width: 24px;
                max-width: 28px;
                padding: 2px 4px;
                border-radius: 3px;
                border: 1px solid transparent;
                background-color: transparent;
                color: #E8E8FF;
            }

            #TitleBar QPushButton#TitleButton:hover {
                background-color: #262247;
            }

            #TitleBar QPushButton#TitleButtonClose:hover {
                background-color: #FF3C68;
                border-color: #FF3C68;
            }
        """


        if theme_key == "light":
            css = light_css
        elif theme_key == "neon":
            css = neon_css
        else:
            css = dark_css

        # Apply a global UI scale by tweaking the base font size.
        # All bundled stylesheets use "font-size: 10pt;" as a starting point.
        try:
            scale = int(getattr(self, "_ui_scale_percent", 100))
        except Exception:
            scale = 100
        scale = max(50, min(200, scale))  # safety clamp

        base_font_pt = 10.0 * scale / 100.0
        font_size_str = f"font-size: {base_font_pt:.1f}pt;"

        css = css.replace("font-size: 10pt;", font_size_str)

        # Make the main tab bar easier to read: slightly larger + heavier font.
        tab_font_pt = base_font_pt + 1.0
        css += f"""
QTabBar::tab {{
    font-size: {tab_font_pt:.1f}pt;
    font-weight: 600;
    padding: 6px 18px;
}}
"""

        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(css)

    def _apply_scale(self, scale_percent: int) -> None:
        """Update UI scale factor and re-apply the current theme stylesheet.

        This is called when the user changes the "UI scale" in the Settings tab.
        """
        try:
            scale = int(scale_percent)
        except Exception:
            scale = 100
        # Clamp to a reasonable range to avoid absurd values.
        scale = max(50, min(200, scale))
        self._ui_scale_percent = scale

        # Re-apply the current theme so the stylesheet is regenerated
        # with the new font sizes.
        self._apply_theme(self._theme_key)


    def update_now_playing(self, _text: str) -> None:
        """
        Update the label in the global player bar.

        We deliberately ignore the raw audio description and show
        the current project name instead, so the user always sees
        which Olaf project is active.
        """
        if self._current_project is None:
            label = "No project selected"
        else:
            label = f"Project: {self._current_project.name}"
        self.now_playing_label.setText(label)

    def _on_global_play_pause_clicked(self) -> None:
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_player_state_changed_global(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.global_play_pause_btn.setText("Pause")
        else:
            self.global_play_pause_btn.setText("Play")

    def _on_player_position_changed_global(self, position: int) -> None:
        self.global_position_slider.blockSignals(True)
        self.global_position_slider.setValue(position)
        self.global_position_slider.blockSignals(False)

    def _on_player_duration_changed_global(self, duration: int) -> None:
        self.global_position_slider.setRange(0, duration)

    def _on_global_slider_moved(self, value: int) -> None:
        self.player.setPosition(value)

def load_custom_fonts() -> None:
    """
    Load all .ttf and .otf font files from the project-level 'fonts'
    directory so that QFontDatabase can see them in the whole application.

    Layout:
      - Project root:  <project_root>/fonts
    """
    here = Path(__file__).resolve()
    project_root = here.parent.parent  # .../Olaf
    fonts_dir = project_root / "fonts"

    if not fonts_dir.is_dir():
        return

    seen: set[Path] = set()
    for pattern in ("*.ttf", "*.otf"):
        for font_path in sorted(fonts_dir.glob(pattern)):
            if font_path in seen:
                continue
            QFontDatabase.addApplicationFont(str(font_path))
            seen.add(font_path)

def main() -> None:
    app = QApplication(sys.argv)

    # Load bundled fonts before creating any windows
    load_custom_fonts()

    win = MainWindow()
    win.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
