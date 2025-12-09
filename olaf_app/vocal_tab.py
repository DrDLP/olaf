from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Any
import json
import shutil
from PyQt6.QtCore import Qt, QUrl, pyqtSignal, QSettings
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QProgressBar,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QCheckBox,
    QSpinBox,          
    QDoubleSpinBox,
    QApplication,
    QTabWidget,
    QSlider,
    QInputDialog,
)




from .project_manager import Project
from .vocal_alignment import run_alignment_for_project

# torch is optional; we handle the case where it is not installed
try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class VocalTab(QWidget):
    """
    Vocal tab:
    - Manages lyrics (stored inside the project)
    - Runs Whisper-based alignment (phrases + words)
    - Lets the user edit phrase timings
    - Saves updated timings to JSON + SRT
    - Provides a simple karaoke preview driven by the shared QMediaPlayer
    """
    audioStarted = pyqtSignal(str)

    def __init__(self, player: QMediaPlayer, parent=None):
        super().__init__(parent)
        self.player = player
        self._project: Optional[Project] = None

        # Alignment data (generic: dicts or dataclasses)
        self._phrases: List[Any] = []
        self._words: List[Any] = []
        
        # Cached media duration in milliseconds (for sliders)
        self._media_duration_ms: int = 0
       

        # GPU detection state
        self._gpu_available: bool = False
        self._gpu_reason: str = ""

        self._build_ui()
        self._detect_gpu()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Project label is now hidden; the active project is shown
        # globally in the bottom player bar.
        self.lbl_project = QLabel("", self)
        self.lbl_project.setVisible(False)
        main_layout.addWidget(self.lbl_project)

        # Sub-tabs inside the Vocal tab
        self.sub_tabs = QTabWidget(self)
        main_layout.addWidget(self.sub_tabs)

        # --------------------------------------------------------------
        # Tab 1: Lyrics alignment (lyrics + alignment settings)
        # --------------------------------------------------------------
        tab_lyrics = QWidget(self)
        tab_lyrics_layout = QVBoxLayout(tab_lyrics)

        # ----- Lyrics group -----
        lyrics_group = QGroupBox("Lyrics", tab_lyrics)
        lg_layout = QVBoxLayout(lyrics_group)

        btn_row = QHBoxLayout()
        self.btn_load_lyrics = QPushButton("Load .txt…", lyrics_group)
        self.btn_save_lyrics = QPushButton("Save .txt…", lyrics_group)
        btn_row.addWidget(self.btn_load_lyrics)
        btn_row.addWidget(self.btn_save_lyrics)
        btn_row.addStretch(1)
        lg_layout.addLayout(btn_row)

        self.lyrics_edit = QTextEdit(lyrics_group)
        self.lyrics_edit.setPlaceholderText(
            "One line per phrase (verse). These lyrics are stored inside the project."
        )
        lg_layout.addWidget(self.lyrics_edit)

        lyrics_group.setLayout(lg_layout)
        tab_lyrics_layout.addWidget(lyrics_group)

        # ----- Alignment settings group -----
        settings_group = QGroupBox("Alignment settings", tab_lyrics)
        sg_layout = QVBoxLayout(settings_group)

        # Model selection
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Whisper model:", settings_group))
        self.model_combo = QComboBox(settings_group)
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.model_combo.setCurrentText("medium")
        model_row.addWidget(self.model_combo)
        model_row.addStretch(1)
        sg_layout.addLayout(model_row)

        # Language
        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel("Language:", settings_group))
        self.lang_combo = QComboBox(settings_group)
        # First entry: auto-detect
        self.lang_combo.addItem("Auto (detect language)", userData=None)
        # Explicit languages (data = whisper language code)
        self.lang_combo.addItem("English (en)", userData="en")
        self.lang_combo.addItem("French (fr)", userData="fr")
        self.lang_combo.addItem("German (de)", userData="de")
        self.lang_combo.setCurrentIndex(0)  # default = auto
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch(1)
        sg_layout.addLayout(lang_row)

        # Audio source for alignment and karaoke
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Audio source:", settings_group))
        self.audio_source_combo = QComboBox(settings_group)
        self.audio_source_combo.addItem("Full mix (project audio)", userData="full_mix")
        self.audio_source_combo.addItem("Manual file (browse…)", userData="manual")
        self.audio_source_combo.setCurrentIndex(0)
        source_row.addWidget(self.audio_source_combo)

        self.btn_browse_manual = QPushButton("Browse…", settings_group)
        self.btn_browse_manual.setEnabled(False)  # enabled only when "manual" is selected
        source_row.addWidget(self.btn_browse_manual)

        self.lbl_manual_path = QLabel("", settings_group)
        self.lbl_manual_path.setMinimumWidth(220)
        self.lbl_manual_path.setWordWrap(False)
        self.lbl_manual_path.setToolTip("Selected manual audio file")
        source_row.addWidget(self.lbl_manual_path, 1)

        source_row.addStretch(1)
        sg_layout.addLayout(source_row)

        # Device (CPU / GPU) + status
        device_row = QHBoxLayout()
        self.chk_use_gpu = QCheckBox("Use GPU if available", settings_group)
        self.lbl_gpu_status = QLabel("", settings_group)
        self.lbl_gpu_status.setWordWrap(True)
        device_row.addWidget(self.chk_use_gpu)
        device_row.addWidget(self.lbl_gpu_status, 1)
        sg_layout.addLayout(device_row)
        
        # Whisper decoding quality (beam search parameters)
        decode_row = QHBoxLayout()
        decode_row.addWidget(QLabel("Beam size:", settings_group))
        self.spin_beam_size = QSpinBox(settings_group)
        self.spin_beam_size.setRange(1, 16)
        self.spin_beam_size.setValue(5)
        self.spin_beam_size.setToolTip(
            "Number of parallel paths explored by Whisper.\n"
            "Higher = better quality, but slower decoding."
        )
        decode_row.addWidget(self.spin_beam_size)

        decode_row.addWidget(QLabel("Patience:", settings_group))
        self.spin_patience = QDoubleSpinBox(settings_group)
        self.spin_patience.setDecimals(2)
        self.spin_patience.setRange(0.0, 2.0)
        self.spin_patience.setSingleStep(0.1)
        self.spin_patience.setValue(1.0)
        self.spin_patience.setToolTip(
            "Beam search patience factor.\n"
            "Values > 1.0 let Whisper explore more candidates before stopping."
        )
        decode_row.addWidget(self.spin_patience)
        decode_row.addStretch(1)
        sg_layout.addLayout(decode_row)

        # Alignment refinement parameters (lyrics ↔ recognized words)
        align_params_row = QHBoxLayout()
        align_params_row.addWidget(QLabel("Max search window:", settings_group))
        self.spin_max_search_window = QSpinBox(settings_group)
        self.spin_max_search_window.setRange(1, 50)
        self.spin_max_search_window.setValue(5)
        self.spin_max_search_window.setToolTip(
            "How many recognized words to look ahead when aligning each lyrics word.\n"
            "Larger window is more robust to insertions but slightly slower."
        )
        align_params_row.addWidget(self.spin_max_search_window)

        align_params_row.addWidget(QLabel("Min similarity:", settings_group))
        self.spin_min_similarity = QDoubleSpinBox(settings_group)
        self.spin_min_similarity.setDecimals(2)
        self.spin_min_similarity.setRange(0.0, 1.0)
        self.spin_min_similarity.setSingleStep(0.05)
        self.spin_min_similarity.setValue(0.60)
        self.spin_min_similarity.setToolTip(
            "Minimal similarity (0–1) between lyrics word and recognized word.\n"
            "Higher = stricter matches (fewer false positives, more 'unmatched' words)."
        )
        align_params_row.addWidget(self.spin_min_similarity)
        align_params_row.addStretch(1)
        sg_layout.addLayout(align_params_row)

        # --------------------------------------------------------------
        # Advanced Whisper options (with presets)
        # --------------------------------------------------------------
        adv_group = QGroupBox("Whisper – advanced options", settings_group)
        adv_layout = QVBoxLayout(adv_group)

        # Presets row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:", adv_group))
        self.combo_whisper_preset = QComboBox(adv_group)
        # Internal keys: "balanced", "harsh", "custom"
        self.combo_whisper_preset.addItem("Balanced music (default)", userData="balanced")
        self.combo_whisper_preset.addItem("Harsh / noisy vocals", userData="harsh")
        self.combo_whisper_preset.addItem("Custom (manual tuning)", userData="custom")
        self.combo_whisper_preset.setCurrentIndex(0)
        self.combo_whisper_preset.setToolTip(
            "Balanced: good default for most songs.\n"
            "Harsh / noisy: more tolerant to growls, screams, noisy mixes,\n"
            "and multilingual/Latin content, at the cost of more noise."
        )
        preset_row.addWidget(self.combo_whisper_preset)
        preset_row.addStretch(1)
        adv_layout.addLayout(preset_row)

        # Row: no_speech_threshold + condition_on_previous_text
        row_ns = QHBoxLayout()
        row_ns.addWidget(QLabel("No-speech threshold:", adv_group))
        self.spin_no_speech_threshold = QDoubleSpinBox(adv_group)
        self.spin_no_speech_threshold.setDecimals(2)
        self.spin_no_speech_threshold.setRange(0.0, 1.0)
        self.spin_no_speech_threshold.setSingleStep(0.05)
        self.spin_no_speech_threshold.setValue(0.60)
        self.spin_no_speech_threshold.setToolTip(
            "Whisper's threshold to classify segments as 'no speech'.\n"
            "Lower values keep more low-intelligibility segments\n"
            "(useful for harsh vocals), but may include more noise."
        )
        row_ns.addWidget(self.spin_no_speech_threshold)

        self.chk_condition_prev = QCheckBox("Use previous text as context", adv_group)
        self.chk_condition_prev.setChecked(True)
        self.chk_condition_prev.setToolTip(
            "If enabled, each segment is decoded using the previous text as context.\n"
            "Disable this for strongly multilingual or Latin content to reduce\n"
            "language drift back to English/French."
        )
        row_ns.addWidget(self.chk_condition_prev)
        row_ns.addStretch(1)
        adv_layout.addLayout(row_ns)

        # Row: compression_ratio_threshold + logprob_threshold
        row_filters = QHBoxLayout()
        row_filters.addWidget(QLabel("Compression ratio max:", adv_group))
        self.spin_compression_ratio = QDoubleSpinBox(adv_group)
        self.spin_compression_ratio.setDecimals(2)
        self.spin_compression_ratio.setRange(0.0, 10.0)
        self.spin_compression_ratio.setSingleStep(0.1)
        self.spin_compression_ratio.setValue(2.40)
        self.spin_compression_ratio.setToolTip(
            "Filter for hallucinated / repetitive segments.\n"
            "Higher values are more tolerant (fewer segments dropped)."
        )
        row_filters.addWidget(self.spin_compression_ratio)

        row_filters.addWidget(QLabel("Log-prob threshold:", adv_group))
        self.spin_logprob_threshold = QDoubleSpinBox(adv_group)
        self.spin_logprob_threshold.setDecimals(2)
        self.spin_logprob_threshold.setRange(-10.0, 0.0)
        self.spin_logprob_threshold.setSingleStep(0.1)
        self.spin_logprob_threshold.setValue(-1.00)
        self.spin_logprob_threshold.setToolTip(
            "Average log-probability threshold for rejecting segments.\n"
            "Lower (more negative) = more tolerant (keeps more uncertain segments)."
        )
        row_filters.addWidget(self.spin_logprob_threshold)
        row_filters.addStretch(1)
        adv_layout.addLayout(row_filters)

        # Initial prompt for Latin / domain-specific vocabulary
        prompt_label = QLabel("Initial prompt (optional):", adv_group)
        prompt_label.setToolTip(
            "Optional text shown to Whisper before decoding.\n"
            "Can be used to hint the model about Latin / specific vocabulary,\n"
            "e.g. 'Ecclesiastical Latin liturgical lyrics'."
        )
        adv_layout.addWidget(prompt_label)

        self.txt_initial_prompt = QTextEdit(adv_group)
        self.txt_initial_prompt.setPlaceholderText(
            "Example: 'Latin liturgical lyrics, ecclesiastical pronunciation.'"
        )
        self.txt_initial_prompt.setFixedHeight(60)
        adv_layout.addWidget(self.txt_initial_prompt)

        adv_help = QLabel(
            "Balanced preset: good starting point for most songs.\n"
            "Harsh / noisy: more tolerant settings for death metal, growls,\n"
            "and multilingual/Latin content – at the cost of more noise.",
            adv_group,
        )
        adv_help.setWordWrap(True)
        adv_layout.addWidget(adv_help)

        adv_group.setLayout(adv_layout)
        sg_layout.addWidget(adv_group)

        # Apply default Whisper preset and connect changes
        self._apply_whisper_preset("balanced")
        self.combo_whisper_preset.currentIndexChanged.connect(self._on_whisper_preset_changed)
        
        # Run alignment
        align_row = QHBoxLayout()
        self.btn_align = QPushButton("Align lyrics with audio", settings_group)
        align_row.addWidget(self.btn_align)
        align_row.addStretch(1)
        sg_layout.addLayout(align_row)

        self.align_progress = QProgressBar(settings_group)
        self.align_progress.setRange(0, 100)
        self.align_progress.setValue(0)
        self.lbl_align_status = QLabel("Ready.", settings_group)

        sg_layout.addWidget(self.align_progress)
        sg_layout.addWidget(self.lbl_align_status)

        settings_group.setLayout(sg_layout)
        tab_lyrics_layout.addWidget(settings_group)
        tab_lyrics_layout.addStretch(1)

        tab_lyrics.setLayout(tab_lyrics_layout)

        # --------------------------------------------------------------
        # Tab 2: Export (ancien Alignment management caché)
        # --------------------------------------------------------------
        tab_manage = QWidget(self)
        tab_manage_layout = QVBoxLayout(tab_manage)

        # Ancien preview (toujours là pour la logique, mais caché)
        preview_group = QGroupBox("Alignment preview", tab_manage)
        pg_layout = QVBoxLayout(preview_group)

        main_row = QHBoxLayout()

        # Left: list of segments
        self.segments_list = QListWidget(preview_group)
        main_row.addWidget(self.segments_list, 1)

        # Right: timing editor
        editor_col = QVBoxLayout()

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Start (s):", preview_group))
        self.spin_start = QDoubleSpinBox(preview_group)
        self.spin_start.setRange(0.0, 100000.0)
        self.spin_start.setDecimals(3)
        time_row.addWidget(self.spin_start)

        time_row.addWidget(QLabel("End (s):", preview_group))
        self.spin_end = QDoubleSpinBox(preview_group)
        self.spin_end.setRange(0.0, 100000.0)
        self.spin_end.setDecimals(3)
        time_row.addWidget(self.spin_end)

        editor_col.addLayout(time_row)

        self.btn_apply_timing = QPushButton("Apply to selected", preview_group)
        editor_col.addWidget(self.btn_apply_timing, alignment=Qt.AlignmentFlag.AlignLeft)

        self.btn_save_timings = QPushButton("Save timings to JSON + SRT", preview_group)
        editor_col.addWidget(self.btn_save_timings, alignment=Qt.AlignmentFlag.AlignLeft)

        main_row.addLayout(editor_col, 1)
        pg_layout.addLayout(main_row)

        preview_group.setLayout(pg_layout)
        tab_manage_layout.addWidget(preview_group)

        # On garde ce group pour la logique mais on le cache dans l'onglet Export
        preview_group.hide()

        # ---- Export / Import group ----
        export_group = QGroupBox("Export / Import alignment files", tab_manage)
        export_layout = QVBoxLayout(export_group)

        export_info = QLabel(
            "Export or import alignment in several formats:\n"
            "- Subtitles (SRT) for YouTube / DaVinci Resolve\n"
            "- Phrase-level timings (JSON)\n"
            "- Word-level timings (JSON)\n"
            "- Phoneme-level timings (JSON)\n"
            "- Plain lyrics text (TXT)",
            export_group,
        )
        export_info.setWordWrap(True)
        export_layout.addWidget(export_info)

        # ---- SRT ----
        row_srt = QHBoxLayout()
        row_srt.addWidget(QLabel("Subtitles (SRT):", export_group))
        self.btn_export_srt = QPushButton("Export SRT…", export_group)
        self.btn_import_srt = QPushButton("Import SRT…", export_group)
        row_srt.addWidget(self.btn_export_srt)
        row_srt.addWidget(self.btn_import_srt)
        row_srt.addStretch(1)
        export_layout.addLayout(row_srt)

        # ---- Phrases ----
        row_phr = QHBoxLayout()
        row_phr.addWidget(QLabel("Phrase timings:", export_group))
        self.btn_export_phrases_json = QPushButton("Export JSON…", export_group)
        self.btn_import_phrases_json = QPushButton("Import JSON…", export_group)
        row_phr.addWidget(self.btn_export_phrases_json)
        row_phr.addWidget(self.btn_import_phrases_json)
        row_phr.addStretch(1)
        export_layout.addLayout(row_phr)

        # ---- Words ----
        row_words = QHBoxLayout()
        row_words.addWidget(QLabel("Word timings:", export_group))
        self.btn_export_words_json = QPushButton("Export JSON…", export_group)
        self.btn_import_words_json = QPushButton("Import JSON…", export_group)
        row_words.addWidget(self.btn_export_words_json)
        row_words.addWidget(self.btn_import_words_json)
        row_words.addStretch(1)
        export_layout.addLayout(row_words)

        # ---- Phonemes ----
        row_phon = QHBoxLayout()
        row_phon.addWidget(QLabel("Phoneme timings:", export_group))
        self.btn_export_phonemes_json = QPushButton("Export JSON…", export_group)
        self.btn_import_phonemes_json = QPushButton("Import JSON…", export_group)
        row_phon.addWidget(self.btn_export_phonemes_json)
        row_phon.addWidget(self.btn_import_phonemes_json)
        row_phon.addStretch(1)
        export_layout.addLayout(row_phon)

        # ---- Lyrics TXT ----
        row_txt = QHBoxLayout()
        row_txt.addWidget(QLabel("Lyrics text:", export_group))
        self.btn_export_lyrics_txt = QPushButton("Export TXT…", export_group)
        self.btn_import_lyrics_txt = QPushButton("Import TXT…", export_group)
        row_txt.addWidget(self.btn_export_lyrics_txt)
        row_txt.addWidget(self.btn_import_lyrics_txt)
        row_txt.addStretch(1)
        export_layout.addLayout(row_txt)

        # ---- Status + bulk export ----
        self.lbl_export_status = QLabel("", export_group)
        export_layout.addWidget(self.lbl_export_status)

        self.btn_export_all = QPushButton("Export all files…", export_group)
        export_layout.addWidget(self.btn_export_all)

        tab_manage_layout.addWidget(export_group)
        tab_manage_layout.addStretch(1)
        tab_manage.setLayout(tab_manage_layout)


        # --------------------------------------------------------------
        # Tab 3: Karaoke preview (Alignment management (karaoke preview))
        # --------------------------------------------------------------
        tab_kara = QWidget(self)
        tab_kara_layout = QVBoxLayout(tab_kara)

        kara_group = QGroupBox("Karaoke preview", tab_kara)
        kara_layout = QHBoxLayout(kara_group)

        # Left column: current line + word + local player + phrase + word timing
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(4)  # compact vertical spacing

        # Current phrase and current word
        self.lbl_kara_line = QLabel("", kara_group)
        self.lbl_kara_word = QLabel("", kara_group)
        self.lbl_kara_line.setStyleSheet("font-weight: bold;")
        self.lbl_kara_word.setStyleSheet("color: #00ccff;")

        left_col.addWidget(self.lbl_kara_line)
        left_col.addWidget(self.lbl_kara_word)

        # Timer label and position slider for the current audio
        self.lbl_kara_track_time = QLabel("0.000 s", kara_group)
        left_col.addWidget(self.lbl_kara_track_time)

        self.kara_slider = QSlider(Qt.Orientation.Horizontal, kara_group)
        self.kara_slider.setRange(0, 0)

        # Neon-style fake waveform background for the karaoke track slider
        self.kara_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 18px;
                margin: 0px;
                border-radius: 9px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0.00 #050008,
                    stop: 0.15 #200020,
                    stop: 0.30 #ff00ff,
                    stop: 0.45 #300030,
                    stop: 0.60 #ff66ff,
                    stop: 0.75 #300030,
                    stop: 0.90 #ff00ff,
                    stop: 1.00 #050008
                );
            }
            QSlider::handle:horizontal {
                background: #000000;
                border: 1px solid #ff66ff;
                width: 10px;
                margin: -6px 0;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 255, 255, 120);
                border-radius: 9px;
            }
            QSlider::add-page:horizontal {
                background: transparent;
                border-radius: 9px;
            }
            """
        )

        left_col.addWidget(self.kara_slider)

        # Local player controls (Play / Pause / Stop) for the selected audio source
        kara_controls = QHBoxLayout()
        self.btn_karaoke_play = QPushButton("Play", kara_group)
        self.btn_karaoke_pause = QPushButton("Pause", kara_group)
        self.btn_karaoke_stop = QPushButton("Stop", kara_group)
        kara_controls.addWidget(self.btn_karaoke_play)
        kara_controls.addWidget(self.btn_karaoke_pause)
        kara_controls.addWidget(self.btn_karaoke_stop)
        kara_controls.addStretch(1)
        left_col.addLayout(kara_controls)

        # Sliders to adjust the start/end of the selected phrase
        phrase_group = QGroupBox("Selected phrase timing", kara_group)
        phrase_layout = QVBoxLayout(phrase_group)

        self.lbl_phrase_start_value = QLabel("Start: 0.000 s", phrase_group)
        self.phrase_start_slider = QSlider(Qt.Orientation.Horizontal, phrase_group)
        self.phrase_start_slider.setRange(0, 0)

        self.lbl_phrase_end_value = QLabel("End: 0.000 s", phrase_group)
        self.phrase_end_slider = QSlider(Qt.Orientation.Horizontal, phrase_group)
        self.phrase_end_slider.setRange(0, 0)

        # 10 ms nudge buttons for phrase start
        phrase_start_row = QHBoxLayout()
        phrase_start_row.addWidget(self.lbl_phrase_start_value)
        self.btn_phrase_start_minus = QPushButton("-10 ms", phrase_group)
        self.btn_phrase_start_plus = QPushButton("+10 ms", phrase_group)
        phrase_start_row.addWidget(self.btn_phrase_start_minus)
        phrase_start_row.addWidget(self.btn_phrase_start_plus)
        phrase_start_row.addStretch(1)
        phrase_layout.addLayout(phrase_start_row)
        phrase_layout.addWidget(self.phrase_start_slider)

        # 10 ms nudge buttons for phrase end
        phrase_end_row = QHBoxLayout()
        phrase_end_row.addWidget(self.lbl_phrase_end_value)
        self.btn_phrase_end_minus = QPushButton("-10 ms", phrase_group)
        self.btn_phrase_end_plus = QPushButton("+10 ms", phrase_group)
        phrase_end_row.addWidget(self.btn_phrase_end_minus)
        phrase_end_row.addWidget(self.btn_phrase_end_plus)
        phrase_end_row.addStretch(1)
        phrase_layout.addLayout(phrase_end_row)
        phrase_layout.addWidget(self.phrase_end_slider)

        left_col.addWidget(phrase_group)


        # Sliders to adjust the start/end of the selected word + insert/delete
        word_group = QGroupBox("Selected word timing", kara_group)
        wg_layout = QVBoxLayout(word_group)

        self.lbl_word_text = QLabel("(no word selected)", word_group)
        wg_layout.addWidget(self.lbl_word_text)

        self.lbl_word_start_value = QLabel("Start: 0.000 s", word_group)
        self.word_start_slider = QSlider(Qt.Orientation.Horizontal, word_group)
        self.word_start_slider.setRange(0, 0)

        self.lbl_word_end_value = QLabel("End: 0.000 s", word_group)
        self.word_end_slider = QSlider(Qt.Orientation.Horizontal, word_group)
        self.word_end_slider.setRange(0, 0)

        # 10 ms nudge buttons for word start
        word_start_row = QHBoxLayout()
        word_start_row.addWidget(self.lbl_word_start_value)
        self.btn_word_start_minus = QPushButton("-10 ms", word_group)
        self.btn_word_start_plus = QPushButton("+10 ms", word_group)
        word_start_row.addWidget(self.btn_word_start_minus)
        word_start_row.addWidget(self.btn_word_start_plus)
        word_start_row.addStretch(1)
        wg_layout.addLayout(word_start_row)
        wg_layout.addWidget(self.word_start_slider)

        # 10 ms nudge buttons for word end
        word_end_row = QHBoxLayout()
        word_end_row.addWidget(self.lbl_word_end_value)
        self.btn_word_end_minus = QPushButton("-10 ms", word_group)
        self.btn_word_end_plus = QPushButton("+10 ms", word_group)
        word_end_row.addWidget(self.btn_word_end_minus)
        word_end_row.addWidget(self.btn_word_end_plus)
        word_end_row.addStretch(1)
        wg_layout.addLayout(word_end_row)
        wg_layout.addWidget(self.word_end_slider)

        word_btn_row = QHBoxLayout()
        self.btn_insert_word = QPushButton("Insert word", word_group)
        self.btn_delete_word = QPushButton("Delete word", word_group)
        word_btn_row.addWidget(self.btn_insert_word)
        word_btn_row.addWidget(self.btn_delete_word)
        word_btn_row.addStretch(1)
        wg_layout.addLayout(word_btn_row)

        word_group.setLayout(wg_layout)
        left_col.addWidget(word_group)


        # Wrap left column in a QWidget so we can align it to the top
        left_widget = QWidget(kara_group)
        left_widget.setLayout(left_col)

        # Center column: phrases list
        self.kara_scroll_list = QListWidget(kara_group)
        # Use default palette / theme for background and selection

        # Right column: words of the selected/current phrase
        self.kara_words_list = QListWidget(kara_group)
        # Use default palette / theme for background and selection


        # Use a very soft alternating background so items stay readable
        # on dark / neon themes without being too aggressive.
        subtle_list_style = """
            QListWidget {
                padding-left: 10px;
                padding-right: 10px;
                border: none;
                background-color: transparent;
            }
            QListWidget::item {
                padding: 2px 6px;
                background-color: transparent;
            }
            /* Slightly lighter background for every other row */
            QListWidget::item:alternate {
                background-color: rgba(255, 255, 255, 18);
            }
            /* Keep selection clearly visible whatever the theme */
            QListWidget::item:selected {
                background-color: rgba(70, 194, 255, 150);
                color: #ffffff;
            }
        """

        self.kara_scroll_list.setStyleSheet(subtle_list_style)
        self.kara_words_list.setStyleSheet(subtle_list_style)


        # Add widgets to main horizontal layout
        kara_layout.addWidget(left_widget)
        kara_layout.addWidget(self.kara_scroll_list)
        kara_layout.addWidget(self.kara_words_list)

        # Force left column to be top-aligned inside the horizontal layout
        kara_layout.setAlignment(left_widget, Qt.AlignmentFlag.AlignTop)

        # 3 columns with equal width
        kara_layout.setStretch(0, 1)  # left column
        kara_layout.setStretch(1, 1)  # phrases
        kara_layout.setStretch(2, 1)  # words

        kara_group.setLayout(kara_layout)
        tab_kara_layout.addWidget(kara_group)
        tab_kara.setLayout(tab_kara_layout)

        # --------------------------------------------------------------
        # Add tabs in desired order
        # --------------------------------------------------------------
        self.sub_tabs.addTab(tab_lyrics, "Lyrics alignment")
        self.sub_tabs.addTab(tab_kara, "Alignement management (karaoke preview)")
        self.sub_tabs.addTab(tab_manage, "Export / Import")

        # --------------------------------------------------------------
        # Connections at the end of UI setup
        # --------------------------------------------------------------
        self.btn_load_lyrics.clicked.connect(self.load_lyrics_from_file)
        self.btn_save_lyrics.clicked.connect(self.save_lyrics_to_file)
        self.btn_align.clicked.connect(self.run_alignment)

        self.segments_list.currentItemChanged.connect(self.on_segment_selected)
        self.btn_apply_timing.clicked.connect(self.apply_timing_to_selected)
        self.btn_save_timings.clicked.connect(self.save_timings_to_files)

        # Export: individual files
        self.btn_export_srt.clicked.connect(self.export_srt)
        self.btn_export_phrases_json.clicked.connect(self.export_phrases_json)
        self.btn_export_words_json.clicked.connect(self.export_words_json)
        self.btn_export_phonemes_json.clicked.connect(self.export_phonemes_json)
        self.btn_export_lyrics_txt.clicked.connect(self.export_lyrics_txt)

        # Import: individual files (JSON / TXT / SRT)
        self.btn_import_srt.clicked.connect(self.import_srt)
        self.btn_import_phrases_json.clicked.connect(self.import_phrases_json)
        self.btn_import_words_json.clicked.connect(self.import_words_json)
        self.btn_import_phonemes_json.clicked.connect(self.import_phonemes_json)
        self.btn_import_lyrics_txt.clicked.connect(self.import_lyrics_txt)

        # Export: bulk
        self.btn_export_all.clicked.connect(self.export_alignment_files)


        # Karaoke controls
        self.btn_karaoke_play.clicked.connect(self.play_karaoke)
        self.btn_karaoke_pause.clicked.connect(self.pause_karaoke)
        self.btn_karaoke_stop.clicked.connect(self.stop_karaoke)
        self.kara_slider.sliderMoved.connect(self._on_kara_slider_moved)

        # Phrase timing sliders (edit selected phrase)
        self.kara_scroll_list.currentItemChanged.connect(self._on_kara_phrase_selected)
        self.phrase_start_slider.sliderMoved.connect(self._on_phrase_start_slider_moved)
        self.phrase_end_slider.sliderMoved.connect(self._on_phrase_end_slider_moved)
        self.phrase_start_slider.sliderReleased.connect(self._on_phrase_sliders_released)
        self.phrase_end_slider.sliderReleased.connect(self._on_phrase_sliders_released)

        # Word timing sliders (edit selected word)
        self.kara_words_list.currentItemChanged.connect(self._on_kara_word_selected)
        self.word_start_slider.sliderMoved.connect(self._on_word_start_slider_moved)
        self.word_end_slider.sliderMoved.connect(self._on_word_end_slider_moved)
        self.word_start_slider.sliderReleased.connect(self._on_word_sliders_released)
        self.word_end_slider.sliderReleased.connect(self._on_word_sliders_released)
        self.btn_insert_word.clicked.connect(self.insert_word_for_current_phrase)
        self.btn_delete_word.clicked.connect(self.delete_selected_word)
        
        # Nudge buttons (10 ms)
        self.btn_phrase_start_minus.clicked.connect(self._nudge_phrase_start_minus)
        self.btn_phrase_start_plus.clicked.connect(self._nudge_phrase_start_plus)
        self.btn_phrase_end_minus.clicked.connect(self._nudge_phrase_end_minus)
        self.btn_phrase_end_plus.clicked.connect(self._nudge_phrase_end_plus)

        self.btn_word_start_minus.clicked.connect(self._nudge_word_start_minus)
        self.btn_word_start_plus.clicked.connect(self._nudge_word_start_plus)
        self.btn_word_end_minus.clicked.connect(self._nudge_word_end_minus)
        self.btn_word_end_plus.clicked.connect(self._nudge_word_end_plus)
        

        # Extra connections for audio source + persistence
        self.audio_source_combo.currentIndexChanged.connect(self._on_audio_source_changed)
        self.btn_browse_manual.clicked.connect(self._browse_manual_audio)

        # Persist lyrics as the user types
        self.lyrics_edit.textChanged.connect(self._save_persisted_lyrics)

        # Load persisted settings (lyrics text, audio source mode, manual path)
        self._load_persisted_prefs()

    # ------------------------------------------------------------------
    # Whisper presets (balanced / harsh / custom)
    # ------------------------------------------------------------------

    def _on_whisper_preset_changed(self) -> None:
        """React to preset combo changes and update advanced parameters."""
        idx = self.combo_whisper_preset.currentIndex()
        key = self.combo_whisper_preset.itemData(idx)
        if not isinstance(key, str):
            return
        if key == "custom":
            # Do not override manual tuning in custom mode.
            return
        self._apply_whisper_preset(key)

    def _apply_whisper_preset(self, preset_key: str) -> None:
        """
        Apply a preset for Whisper + alignment parameters.

        'balanced' is suited for most songs.
        'harsh' is more tolerant to extreme vocals / noisy mixes.
        """
        # Safe defaults in case some widgets are missing
        if not hasattr(self, "spin_beam_size"):
            return

        if preset_key == "harsh":
            # More exhaustive search + more tolerant filters
            self.spin_beam_size.setValue(8)
            self.spin_patience.setValue(1.20)

            self.spin_no_speech_threshold.setValue(0.35)
            self.spin_compression_ratio.setValue(3.00)
            self.spin_logprob_threshold.setValue(-1.50)

            self.chk_condition_prev.setChecked(False)

            # Alignment: more tolerant to recognition errors
            self.spin_max_search_window.setValue(8)
            self.spin_min_similarity.setValue(0.55)
        else:
            # 'balanced' (default) preset
            self.spin_beam_size.setValue(5)
            self.spin_patience.setValue(1.00)

            self.spin_no_speech_threshold.setValue(0.60)
            self.spin_compression_ratio.setValue(2.40)
            self.spin_logprob_threshold.setValue(-1.00)

            self.chk_condition_prev.setChecked(True)

            # Alignment: stricter but cleaner
            self.spin_max_search_window.setValue(5)
            self.spin_min_similarity.setValue(0.60)

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _detect_gpu(self):
        """Detect if CUDA / GPU is usable and explain why / why not."""
        if torch is None:
            self._gpu_available = False
            self._gpu_reason = "PyTorch is not installed in this environment."
            self.chk_use_gpu.setChecked(False)
            self.chk_use_gpu.setEnabled(False)
            self.lbl_gpu_status.setText(self._gpu_reason)
            return

        try:
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self._gpu_available = True
                self._gpu_reason = f"GPU available: {name}"
                self.chk_use_gpu.setEnabled(True)
                self.chk_use_gpu.setChecked(True)
                self.lbl_gpu_status.setText(self._gpu_reason)
            else:
                self._gpu_available = False
                # Try to give a meaningful reason
                if getattr(torch.version, "cuda", None) is None:
                    reason = "PyTorch build has no CUDA support (CPU-only install)."
                elif torch.cuda.device_count() == 0:
                    reason = "No CUDA GPU detected (cuda device_count == 0)."
                else:
                    reason = "CUDA runtime not available (drivers/toolkit issue)."
                self._gpu_reason = reason
                self.chk_use_gpu.setChecked(False)
                # You may want to allow forcing GPU later; for now disable
                self.chk_use_gpu.setEnabled(False)
                self.lbl_gpu_status.setText(f"GPU not available: {reason}")
        except Exception as e:  # pragma: no cover
            self._gpu_available = False
            self._gpu_reason = f"Could not query CUDA: {e}"
            self.chk_use_gpu.setChecked(False)
            self.chk_use_gpu.setEnabled(False)
            self.lbl_gpu_status.setText(self._gpu_reason)

    # ------------------------------------------------------------------
    # Public API used by MainWindow
    # ------------------------------------------------------------------

    def set_project(self, project: Optional[Project]):
        """Set the current project whose vocal alignment will be managed."""
        self._project = project
        if project:
            self.lbl_project.setText(f"Current project: {project.name}")
            # Restore lyrics from project if available; otherwise keep current (persisted) lyrics
            text = getattr(project, "lyrics_text", None)
            if text:
                self.lyrics_edit.setPlainText(text)
        else:
            self.lbl_project.setText("Current project: (none)")
            # Do not clear lyrics_edit here: keep last persisted lyrics

        # Reset alignment data and karaoke visuals
        self._phrases = []
        self._words = []
        self.segments_list.clear()
        self.lbl_kara_line.setText("")
        self.lbl_kara_word.setText("")
        if hasattr(self, "kara_scroll_list"):
            self.kara_scroll_list.clear()

        # Try to load existing alignment from disk (if any)
        self._load_alignment_from_disk()


    def on_position_changed(self, position_ms: int):
        """Update karaoke preview and local player UI based on current playback position."""
        # Update local karaoke slider and track timer, regardless of alignment state
        if hasattr(self, "kara_slider"):
            # Set slider range once duration is known
            try:
                duration_ms = self.player.duration()
            except Exception:
                duration_ms = 0

            if duration_ms > 0:
                self._media_duration_ms = duration_ms

            if self.kara_slider.maximum() == 0 and duration_ms > 0:
                self.kara_slider.setRange(0, duration_ms)


            self.kara_slider.blockSignals(True)
            self.kara_slider.setValue(position_ms)
            self.kara_slider.blockSignals(False)

        if hasattr(self, "lbl_kara_track_time"):
            seconds = position_ms / 1000.0
            self.lbl_kara_track_time.setText(f"{seconds:7.3f} s")

        # Then update phrase/word highlighting only if we have alignment data
        if not self._phrases:
            return

        pos_s = position_ms / 1000.0
        self._update_karaoke_for_time(pos_s)


    # ------------------------------------------------------------------
    # Lyrics file management
    # ------------------------------------------------------------------

    def load_lyrics_from_file(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        start_dir = str(self._project.folder)
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Load lyrics (.txt)",
            start_dir,
            "Text files (*.txt);;All files (*.*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not read lyrics file:\n{e}")
            return

        self.lyrics_edit.setPlainText(text)

        # Persist in project (if helper exists)
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(text)
            except Exception:
                self._project.lyrics_text = text
        else:
            self._project.lyrics_text = text

    def save_lyrics_to_file(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        text = self.lyrics_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Empty lyrics", "Lyrics text is empty.")
            return

        default_path = self._project.folder / "lyrics.txt"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save lyrics (.txt)",
            str(default_path),
            "Text files (*.txt);;All files (*.*)",
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            path.write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write lyrics file:\n{e}")
            return

        # Persist into project
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(text)
            except Exception:
                self._project.lyrics_text = text
        else:
            self._project.lyrics_text = text

    def _get_audio_path_for_current_mode(self) -> Optional[Path]:
        """
        Return audio path to use for alignment and karaoke based on current audio source.
        Supported modes:
          - 'full_mix': use the project's main audio file
          - 'manual'  : use the user-picked file
        """
        if not self._project:
            return None

        mode = self.audio_source_combo.currentData() if self.audio_source_combo is not None else "full_mix"

        # Manual file mode: use the path shown in the label
        if mode == "manual":
            manual = self.lbl_manual_path.text().strip()
            if manual:
                p = Path(manual)
                if p.is_file():
                    return p
                return None

        # Default/fallback: full mix from project
        if hasattr(self._project, "get_audio_path"):
            return self._project.get_audio_path()

        return None

        
    # ------------------------------------------------------------------
    # Persistence helpers (per-user, across app launches)
    # ------------------------------------------------------------------

    def _settings(self) -> QSettings:
        """
        Return a QSettings instance. On Windows it stores in the Registry under:
        HKEY_CURRENT_USER\Software\Olaf\OlafApp
        """
        return QSettings("Olaf", "OlafApp")

    def _load_persisted_prefs(self) -> None:
        """Load last used lyrics and audio source settings from QSettings."""
        s = self._settings()

        # Lyrics text
        last_lyrics = s.value("lyrics_alignment/last_lyrics_text", type=str)
        if last_lyrics:
            # Do not emit textChanged while setting programmatically
            block = self.lyrics_edit.blockSignals(True)
            self.lyrics_edit.setPlainText(last_lyrics)
            self.lyrics_edit.blockSignals(block)

        # Audio source mode
        mode = s.value("lyrics_alignment/audio_source_mode", "full_mix", type=str)
        if mode not in ("full_mix", "manual"):
            mode = "full_mix"

        # Set combo index based on userData
        index_to_set = 0
        for i in range(self.audio_source_combo.count()):
            if self.audio_source_combo.itemData(i) == mode:
                index_to_set = i
                break
        self.audio_source_combo.setCurrentIndex(index_to_set)

        # Manual file path (if any)
        manual_path = s.value("lyrics_alignment/manual_audio_path", "", type=str)
        if manual_path:
            self.lbl_manual_path.setText(manual_path)
            self.lbl_manual_path.setToolTip(manual_path)

        # Enable/disable browse according to mode
        self.btn_browse_manual.setEnabled(mode == "manual")

    def _save_persisted_lyrics(self) -> None:
        """Save current lyrics text to QSettings (called on textChanged)."""
        s = self._settings()
        s.setValue("lyrics_alignment/last_lyrics_text", self.lyrics_edit.toPlainText())

    def _on_audio_source_changed(self, idx: int) -> None:
        """Enable browse button only for 'manual' and persist the choice."""
        mode = self.audio_source_combo.currentData()
        self.btn_browse_manual.setEnabled(mode == "manual")
        # Persist the mode
        s = self._settings()
        s.setValue("lyrics_alignment/audio_source_mode", mode if mode else "full_mix")

    def _browse_manual_audio(self) -> None:
        """Pick a manual audio file and persist its path."""
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio files (*.wav *.mp3 *.flac *.m4a *.aac);;All files (*.*)",
        )
        if not file_path:
            return
        self.lbl_manual_path.setText(file_path)
        self.lbl_manual_path.setToolTip(file_path)
        s = self._settings()
        s.setValue("lyrics_alignment/manual_audio_path", file_path)

    def _load_alignment_from_disk(self) -> None:
        """
        Load existing alignment (phrases + words) from project/vocal_align/*.json
        if available, so timings survive across sessions.
        """
        if not self._project:
            return

        align_dir = self._project.folder / "vocal_align"
        phrases_path = align_dir / "phrases.json"
        words_path = align_dir / "words.json"

        if not phrases_path.is_file():
            # No previous alignment saved for this project
            return

        try:
            phrases_data = json.loads(phrases_path.read_text(encoding="utf-8"))
            if isinstance(phrases_data, list):
                self._phrases = phrases_data
            else:
                self._phrases = []
        except Exception as e:
            self._phrases = []
            self.lbl_align_status.setText(f"Could not read phrases.json: {e}")
            return

        if words_path.is_file():
            try:
                words_data = json.loads(words_path.read_text(encoding="utf-8"))
                if isinstance(words_data, list):
                    self._words = words_data
                else:
                    self._words = []
            except Exception:
                # If words fail to load, we still keep phrases
                self._words = []
        else:
            self._words = []

        # Refresh UI to reflect loaded alignment
        self.populate_segments_preview()
        self.lbl_align_status.setText("Loaded existing alignment from disk.")
        
    # -------------------- Import helpers --------------------

    def _open_file_dialog(self, title: str, filter_str: str) -> Optional[Path]:
        """Small helper to open a file dialog and return a Path or None if cancelled."""
        file_str, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            filter_str,
        )
        if not file_str:
            return None
        return Path(file_str)


    # ------------------------------------------------------------------
    # Alignment + GPU / CPU selection
    # ------------------------------------------------------------------

    def run_alignment(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        # Check audio according to selected source (full mix or manual file)
        audio_path = self._get_audio_path_for_current_mode()
        if not audio_path or not Path(audio_path).is_file():
            mode = "full_mix"
            if hasattr(self, "audio_source_combo") and self.audio_source_combo is not None:
                data = self.audio_source_combo.currentData()
                if isinstance(data, str) and data:
                    mode = data

            if mode == "manual":
                QMessageBox.warning(
                    self,
                    "Invalid audio file",
                    "No valid manual audio file selected.\n"
                    "Please click 'Browse…' or switch the audio source to 'Full mix (project audio)'.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "No audio file",
                    "This project has no valid audio file associated.",
                )
            return



        # Lyrics
        lyrics_text = self.lyrics_edit.toPlainText().strip()
        if not lyrics_text:
            QMessageBox.warning(self, "No lyrics", "Please enter or load lyrics first.")
            return

        # Persist lyrics in the project + to disk so the backend is forced to use them
        if hasattr(self._project, "set_lyrics_text"):
            try:
                self._project.set_lyrics_text(lyrics_text)
            except Exception:
                self._project.lyrics_text = lyrics_text
        else:
            self._project.lyrics_text = lyrics_text

        # Write lyrics.txt in both project root and vocal_align/ for safety
        main_lyrics = self._project.folder / "lyrics.txt"
        align_dir = self._project.folder / "vocal_align"
        align_dir.mkdir(parents=True, exist_ok=True)
        align_lyrics = align_dir / "lyrics.txt"
        for p in (main_lyrics, align_lyrics):
            try:
                p.write_text(lyrics_text, encoding="utf-8")
            except Exception:
                pass

        model_name = self.model_combo.currentText()

        # Use combo userData to decide if we force a language or let Whisper auto-detect
        lang_data = self.lang_combo.currentData()
        if not lang_data:  # Auto (detect language)
            whisper_language = None
        else:
            whisper_language = str(lang_data)

        device = "cuda" if (self.chk_use_gpu.isChecked() and self._gpu_available) else "cpu"

        # Advanced decoding / alignment parameters from the UI
        beam_size = int(self.spin_beam_size.value()) if hasattr(self, "spin_beam_size") else 5
        patience = float(self.spin_patience.value()) if hasattr(self, "spin_patience") else 1.0
        max_search_window = int(self.spin_max_search_window.value()) if hasattr(self, "spin_max_search_window") else 5
        min_similarity = float(self.spin_min_similarity.value()) if hasattr(self, "spin_min_similarity") else 0.60

        # Advanced Whisper filters / context / prompt
        no_speech_threshold = (
            float(self.spin_no_speech_threshold.value())
            if hasattr(self, "spin_no_speech_threshold")
            else None
        )
        compression_ratio_threshold = (
            float(self.spin_compression_ratio.value())
            if hasattr(self, "spin_compression_ratio")
            else None
        )
        logprob_threshold = (
            float(self.spin_logprob_threshold.value())
            if hasattr(self, "spin_logprob_threshold")
            else None
        )
        condition_on_previous_text = (
            bool(self.chk_condition_prev.isChecked())
            if hasattr(self, "chk_condition_prev")
            else None
        )

        initial_prompt = None
        if hasattr(self, "txt_initial_prompt"):
            txt = self.txt_initial_prompt.toPlainText().strip()
            if txt:
                initial_prompt = txt

        self.btn_align.setEnabled(False)
        self.align_progress.setValue(0)

        self.lbl_align_status.setText(f"Running alignment on {device}…")
        QApplication.processEvents()

        def progress_cb(percent: float, message: str):
            self.align_progress.setValue(int(percent))
            self.lbl_align_status.setText(message[:150])
            QApplication.processEvents()

        try:
            # run_alignment_for_project(
            #     project,
            #     audio_path,
            #     lyrics_text,
            #     model_name,
            #     whisper_language,
            #     phoneme_language,
            #     device,
            #     progress_cb,
            # )
            result = run_alignment_for_project(
                project=self._project,
                audio_path=audio_path,
                lyrics_text=lyrics_text,
                model_name=model_name,
                whisper_language=whisper_language,  # None => auto-detect
                phoneme_language=None,
                device=device,
                beam_size=beam_size,
                patience=patience,
                max_search_window=max_search_window,
                min_similarity=min_similarity,
                no_speech_threshold=no_speech_threshold,
                compression_ratio_threshold=compression_ratio_threshold,
                logprob_threshold=logprob_threshold,
                condition_on_previous_text=condition_on_previous_text,
                initial_prompt=initial_prompt,
                progress_cb=progress_cb,
            )



        except Exception as e:
            QMessageBox.critical(self, "Alignment error", f"Whisper / alignment failed:\n{e}")
            self.btn_align.setEnabled(True)
            self.lbl_align_status.setText("Error during alignment.")
            return
        finally:
            self.btn_align.setEnabled(True)


        # Normalize alignment result (dict or dataclass)
        self._phrases = getattr(result, "phrases", getattr(result, "lines", [])) or []
        self._words = getattr(result, "words", []) or []

        self.align_progress.setValue(100)
        self.lbl_align_status.setText("Alignment done.")
        self.populate_segments_preview()

    # ------------------------------------------------------------------
    # Phrases list + editing
    # ------------------------------------------------------------------

    def _get_phrase_field(self, phrase: Any, field: str):
        if phrase is None:
            return None
        if isinstance(phrase, dict):
            return phrase.get(field)
        return getattr(phrase, field, None)

    def _set_phrase_field(self, phrase: Any, field: str, value: Any):
        if isinstance(phrase, dict):
            phrase[field] = value
        else:
            setattr(phrase, field, value)

    def _get_word_field(self, word: Any, field: str):
        if word is None:
            return None
        if isinstance(word, dict):
            return word.get(field)
        return getattr(word, field, None)

    def _set_word_field(self, word: Any, field: str, value: Any):
        if isinstance(word, dict):
            word[field] = value
        else:
            setattr(word, field, value)

    def _get_phrase_neighbor_bounds(self, idx: int) -> tuple[float, float]:
        """
        Return (min_s, max_s) allowed for phrase idx, based on previous/next phrases
        and the track duration. Used to prevent phrase overlaps.
        """
        if not self._phrases or idx < 0 or idx >= len(self._phrases):
            return (0.0, max(self._media_duration_ms / 1000.0, 0.01))

        phrase = self._phrases[idx]
        start = self._get_phrase_field(phrase, "start") or 0.0
        end = self._get_phrase_field(phrase, "end") or start

        # Left bound: end of previous phrase (or 0.0)
        min_s = 0.0
        if idx > 0:
            prev = self._phrases[idx - 1]
            prev_end = self._get_phrase_field(prev, "end")
            if prev_end is not None:
                min_s = float(prev_end)

        # Right bound: start of next phrase (or track duration / end+10s)
        if self._media_duration_ms > 0:
            track_s = self._media_duration_ms / 1000.0
        else:
            track_s = max(end, start) + 10.0

        max_s = track_s
        if idx + 1 < len(self._phrases):
            nxt = self._phrases[idx + 1]
            nxt_start = self._get_phrase_field(nxt, "start")
            if nxt_start is not None:
                max_s = float(nxt_start)

        if max_s < min_s + 0.010:
            max_s = min_s + 0.010

        return (min_s, max_s)

    def _get_word_neighbor_bounds(self, line_index: int, word_global_idx: int) -> tuple[float, float]:
        """
        Return (min_s, max_s) allowed for the given word (global index) within a line_index,
        based on previous/next words in that line and the phrase boundaries.
        """
        if line_index is None:
            # Fallback: whole track
            if self._media_duration_ms > 0:
                return (0.0, self._media_duration_ms / 1000.0)
            return (0.0, 9999.0)

        # Collect all words for this line_index
        words_for_line: list[tuple[int, Any]] = []
        for gidx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                words_for_line.append((gidx, w))

        if not words_for_line:
            if self._media_duration_ms > 0:
                return (0.0, self._media_duration_ms / 1000.0)
            return (0.0, 9999.0)

        # Sort by start time
        def _word_sort_key(item: tuple[int, Any]):
            wobj = item[1]
            ws = self._get_word_field(wobj, "start")
            wi = self._get_word_field(wobj, "word_index")
            return (ws if ws is not None else 0.0, wi if wi is not None else 0)

        words_for_line.sort(key=_word_sort_key)

        # Find position of this word in the sorted list
        idx_in_line = None
        for i, (gidx, _) in enumerate(words_for_line):
            if gidx == word_global_idx:
                idx_in_line = i
                break

        # Phrase bounds for this line
        phrase_start = None
        phrase_end = None
        for phrase in self._phrases:
            if self._get_phrase_field(phrase, "line_index") == line_index:
                phrase_start = self._get_phrase_field(phrase, "start")
                phrase_end = self._get_phrase_field(phrase, "end")
                break

        if self._media_duration_ms > 0:
            track_s = self._media_duration_ms / 1000.0
        else:
            track_s = 9999.0

        # Default min/max from phrase or track
        min_s = phrase_start if phrase_start is not None else 0.0
        max_s = phrase_end if phrase_end is not None else track_s

        if idx_in_line is not None:
            # Previous word end as left bound
            if idx_in_line > 0:
                prev_word = words_for_line[idx_in_line - 1][1]
                prev_end = self._get_word_field(prev_word, "end")
                if prev_end is not None:
                    min_s = float(prev_end)
            # Next word start as right bound
            if idx_in_line + 1 < len(words_for_line):
                next_word = words_for_line[idx_in_line + 1][1]
                next_start = self._get_word_field(next_word, "start")
                if next_start is not None:
                    max_s = float(next_start)

        if max_s < min_s + 0.010:
            max_s = min_s + 0.010

        return (min_s, max_s)


    def _on_kara_word_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """When a word is selected in the words list, sync the 'Selected word timing' panel."""
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        text = self._get_word_field(w, "text") or ""
        start = self._get_word_field(w, "start") or 0.0
        end = self._get_word_field(w, "end") or 0.0
        line_index = self._get_word_field(w, "line_index")

        self.lbl_word_text.setText(f"Word: {text}")

        # Determine slider bounds from neighbor words and phrase
        if line_index is not None:
            min_s, max_s = self._get_word_neighbor_bounds(line_index, idx)
        else:
            if self._media_duration_ms > 0:
                min_s, max_s = (0.0, self._media_duration_ms / 1000.0)
            else:
                min_s, max_s = (0.0, max(end, start) + 1.0)

        # Clamp start/end in those bounds
        start = max(min_s, min(float(start), max_s))
        end = max(start + 0.010, min(float(end), max_s))

        # Convert to ms for slider
        min_ms = int(min_s * 1000.0)
        max_ms = int(max_s * 1000.0)

        self.word_start_slider.blockSignals(True)
        self.word_end_slider.blockSignals(True)

        self.word_start_slider.setRange(min_ms, max_ms)
        self.word_end_slider.setRange(min_ms, max_ms)

        self.word_start_slider.setValue(int(start * 1000.0))
        self.word_end_slider.setValue(int(end * 1000.0))

        self.word_start_slider.blockSignals(False)
        self.word_end_slider.blockSignals(False)

        self.lbl_word_start_value.setText(f"Start: {start:7.3f} s")
        self.lbl_word_end_value.setText(f"End:   {end:7.3f} s")


    def _on_word_start_slider_moved(self, position_ms: int) -> None:
        """Update start label while dragging the word start slider."""
        seconds = position_ms / 1000.0
        self.lbl_word_start_value.setText(f"Start: {seconds:7.3f} s")

    def _on_word_end_slider_moved(self, position_ms: int) -> None:
        """Update end label while dragging the word end slider."""
        seconds = position_ms / 1000.0
        self.lbl_word_end_value.setText(f"End:   {seconds:7.3f} s")

    def _on_word_sliders_released(self) -> None:
        """
        When either start/end slider is released, apply the new timings
        to the selected word and save to disk.
        The new timings are clamped between previous/next words for the same line
        so that no overlap is created. The owning phrase is expanded if needed
        but phrase neighbors are left unchanged.
        """
        current = self.kara_words_list.currentItem()
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        line_index = self._get_word_field(w, "line_index")

        # Neighbor-based bounds
        if line_index is not None:
            min_s, max_s = self._get_word_neighbor_bounds(line_index, idx)
        else:
            if self._media_duration_ms > 0:
                min_s, max_s = (0.0, self._media_duration_ms / 1000.0)
            else:
                min_s, max_s = (0.0, 9999.0)

        start_s = self.word_start_slider.value() / 1000.0
        end_s = self.word_end_slider.value() / 1000.0

        # Clamp to neighbor bounds
        if start_s < min_s:
            start_s = min_s
        if end_s > max_s:
            end_s = max_s

        # Ensure end > start
        if end_s <= start_s:
            end_s = min(start_s + 0.010, max_s)
            if end_s <= start_s:
                start_s = max(min_s, max_s - 0.010)
                end_s = max_s

        # Push clamped values back to sliders
        self.word_start_slider.blockSignals(True)
        self.word_end_slider.blockSignals(True)
        self.word_start_slider.setValue(int(start_s * 1000.0))
        self.word_end_slider.setValue(int(end_s * 1000.0))
        self.word_start_slider.blockSignals(False)
        self.word_end_slider.blockSignals(False)

        # Update word timings
        self._set_word_field(w, "start", float(start_s))
        self._set_word_field(w, "end", float(end_s))

        self.lbl_word_start_value.setText(f"Start: {start_s:7.3f} s")
        self.lbl_word_end_value.setText(f"End:   {end_s:7.3f} s")

        # Expand the owning phrase if needed (but do not touch neighbor phrases)
        phrase_idx_for_line: Optional[int] = None
        phrase_obj: Any = None
        if line_index is not None and self._phrases:
            for p_idx, phrase in enumerate(self._phrases):
                if self._get_phrase_field(phrase, "line_index") == line_index:
                    phrase_idx_for_line = p_idx
                    phrase_obj = phrase
                    break

        if phrase_obj is not None:
            pstart = self._get_phrase_field(phrase_obj, "start")
            pend = self._get_phrase_field(phrase_obj, "end")
            changed = False

            if pstart is None or start_s < float(pstart) - 1e-6:
                self._set_phrase_field(phrase_obj, "start", float(start_s))
                changed = True
            if pend is None or end_s > float(pend) + 1e-6:
                self._set_phrase_field(phrase_obj, "end", float(end_s))
                changed = True

            if changed:
                # Refresh phrase lists and keep selection + sliders in sync
                self.populate_segments_preview()
                if phrase_idx_for_line is not None:
                    if 0 <= phrase_idx_for_line < self.segments_list.count():
                        self.segments_list.setCurrentRow(phrase_idx_for_line)
                    if 0 <= phrase_idx_for_line < self.kara_scroll_list.count():
                        self.kara_scroll_list.setCurrentRow(phrase_idx_for_line)
                    self._sync_phrase_timing_panel(phrase_idx_for_line, phrase_obj)

        # Refresh words list for this line (keeps ordering by time)
        if line_index is not None:
            self._refresh_karaoke_words(line_index)

            # Reselect this word (if still present)
            for row in range(self.kara_words_list.count()):
                item = self.kara_words_list.item(row)
                gidx = item.data(Qt.ItemDataRole.UserRole)
                if gidx == idx:
                    self.kara_words_list.setCurrentItem(item)
                    break

        # Persist new timings to JSON (+ SRT via phrases)
        self.lbl_align_status.setText("Updated word timing from sliders.")
        self.save_timings_to_files()


    # --- 10 ms nudge for phrases ---

    def _nudge_phrase_start(self, delta_ms: int) -> None:
        if self.phrase_start_slider.maximum() <= self.phrase_start_slider.minimum():
            return
        new_val = self.phrase_start_slider.value() + delta_ms
        new_val = max(self.phrase_start_slider.minimum(), min(self.phrase_start_slider.maximum(), new_val))
        self.phrase_start_slider.setValue(new_val)
        self._on_phrase_sliders_released()

    def _nudge_phrase_end(self, delta_ms: int) -> None:
        if self.phrase_end_slider.maximum() <= self.phrase_end_slider.minimum():
            return
        new_val = self.phrase_end_slider.value() + delta_ms
        new_val = max(self.phrase_end_slider.minimum(), min(self.phrase_end_slider.maximum(), new_val))
        self.phrase_end_slider.setValue(new_val)
        self._on_phrase_sliders_released()

    def _nudge_phrase_start_minus(self) -> None:
        self._nudge_phrase_start(-10)

    def _nudge_phrase_start_plus(self) -> None:
        self._nudge_phrase_start(+10)

    def _nudge_phrase_end_minus(self) -> None:
        self._nudge_phrase_end(-10)

    def _nudge_phrase_end_plus(self) -> None:
        self._nudge_phrase_end(+10)

    # --- 10 ms nudge for words ---

    def _nudge_word_start(self, delta_ms: int) -> None:
        if self.word_start_slider.maximum() <= self.word_start_slider.minimum():
            return
        new_val = self.word_start_slider.value() + delta_ms
        new_val = max(self.word_start_slider.minimum(), min(self.word_start_slider.maximum(), new_val))
        self.word_start_slider.setValue(new_val)
        self._on_word_sliders_released()

    def _nudge_word_end(self, delta_ms: int) -> None:
        if self.word_end_slider.maximum() <= self.word_end_slider.minimum():
            return
        new_val = self.word_end_slider.value() + delta_ms
        new_val = max(self.word_end_slider.minimum(), min(self.word_end_slider.maximum(), new_val))
        self.word_end_slider.setValue(new_val)
        self._on_word_sliders_released()

    def _nudge_word_start_minus(self) -> None:
        self._nudge_word_start(-10)

    def _nudge_word_start_plus(self) -> None:
        self._nudge_word_start(+10)

    def _nudge_word_end_minus(self) -> None:
        self._nudge_word_end(-10)

    def _nudge_word_end_plus(self) -> None:
        self._nudge_word_end(+10)


    def insert_word_for_current_phrase(self) -> None:
        """Insert a new word into the current phrase (line_index) with manually editable timing."""
        if not self._phrases:
            QMessageBox.warning(self, "No phrases", "No alignment has been run yet.")
            return

        # Need a current phrase selection
        phrase_item = self.kara_scroll_list.currentItem()
        if phrase_item is None:
            QMessageBox.warning(self, "No phrase selected", "Select a phrase in the list first.")
            return

        phrase_idx = phrase_item.data(Qt.ItemDataRole.UserRole)
        if phrase_idx is None or not isinstance(phrase_idx, int):
            return
        if phrase_idx < 0 or phrase_idx >= len(self._phrases):
            return

        phrase = self._phrases[phrase_idx]
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = phrase_idx

        # Ask for the new word text
        text, ok = QInputDialog.getText(self, "Insert word", "New word text:")
        if not ok or not text.strip():
            return
        text = text.strip()

        # Base time from current playback position
        position_ms = 0
        try:
            position_ms = self.player.position()
        except Exception:
            pass
        base_time = position_ms / 1000.0

        pstart = self._get_phrase_field(phrase, "start")
        pend = self._get_phrase_field(phrase, "end")

        # Clamp base_time in phrase bounds if known
        if pstart is not None and pend is not None:
            if base_time < pstart:
                base_time = float(pstart)
            if base_time > pend:
                base_time = float(pend)

        # Ask user for start time (seconds, 3 decimals)
        default_start = float(base_time)
        start_s, ok = QInputDialog.getDouble(
            self,
            "Word start time",
            "Start (seconds):",
            default_start,
            0.0,
            100000.0,
            3,
        )
        if not ok:
            return

        # Default end = start + 0.200, clamped to phrase end if known
        default_end = start_s + 0.200
        if pstart is not None and pend is not None and default_end > pend:
            default_end = float(pend)

        end_s, ok = QInputDialog.getDouble(
            self,
            "Word end time",
            "End (seconds):",
            default_end,
            start_s,
            100000.0,
            3,
        )
        if not ok:
            return

        if end_s <= start_s:
            end_s = start_s + 0.010

        # Determine word_index: append at the end for this line_index
        word_indices = []
        for w in self._words:
            if self._get_word_field(w, "line_index") == line_index:
                wi = self._get_word_field(w, "word_index")
                if wi is not None:
                    word_indices.append(int(wi))
        if word_indices:
            new_word_index = max(word_indices) + 1
        else:
            new_word_index = 0

        new_word = {
            "line_index": int(line_index),
            "word_index": int(new_word_index),
            "text": text,
            "matched_text": text,
            "start": float(start_s),
            "end": float(end_s),
        }

        self._words.append(new_word)
        global_idx = len(self._words) - 1

        # Refresh words list and select the new word (list is sorted by time)
        self._refresh_karaoke_words(line_index)
        for row in range(self.kara_words_list.count()):
            item = self.kara_words_list.item(row)
            gidx = item.data(Qt.ItemDataRole.UserRole)
            if gidx == global_idx:
                self.kara_words_list.setCurrentItem(item)
                break

        self.lbl_align_status.setText(f"Inserted word '{text}' in phrase {phrase_idx + 1}.")
        self.save_timings_to_files()


    def delete_selected_word(self) -> None:
        """Delete the currently selected word from the alignment."""
        current = self.kara_words_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No word selected", "Select a word in the right list first.")
            return

        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._words):
            return

        w = self._words[idx]
        line_index = self._get_word_field(w, "line_index")
        word_index = self._get_word_field(w, "word_index")
        text = self._get_word_field(w, "text") or ""

        # Remove the word
        del self._words[idx]

        # Optionally reindex following word_index for this line_index
        if line_index is not None and word_index is not None:
            for w2 in self._words:
                if self._get_word_field(w2, "line_index") == line_index:
                    wi2 = self._get_word_field(w2, "word_index")
                    if wi2 is not None and wi2 > word_index:
                        self._set_word_field(w2, "word_index", int(wi2) - 1)

        # Refresh words list
        self._refresh_karaoke_words(line_index)

        # Clear word panel
        self.lbl_word_text.setText("(no word selected)")
        self.lbl_word_start_value.setText("Start: 0.000 s")
        self.lbl_word_end_value.setText("End:   0.000 s")
        self.word_start_slider.setRange(0, 0)
        self.word_end_slider.setRange(0, 0)

        self.lbl_align_status.setText(f"Deleted word '{text}'.")
        self.save_timings_to_files()


    def populate_segments_preview(self):
        self.segments_list.clear()

        if not self._phrases:
            placeholder = QListWidgetItem("No aligned phrases yet.")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.segments_list.addItem(placeholder)
            return

        for idx, phrase in enumerate(self._phrases):
            text = self._get_phrase_field(phrase, "text") or ""
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                times = "[no timing]"
            else:
                times = f"[{start:7.3f} → {end:7.3f}]"
            item = QListWidgetItem(f"{idx + 1:03d} {times}  {text}")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.segments_list.addItem(item)
            
        # Also refresh karaoke scrolling list on the right
        self._refresh_karaoke_scroll()            

    def _refresh_karaoke_scroll(self) -> None:
        """Populate the karaoke scrolling list with all phrases."""
        if not hasattr(self, "kara_scroll_list"):
            return

        self.kara_scroll_list.clear()

        if not self._phrases:
            return

        for idx, phrase in enumerate(self._phrases):
            text = self._get_phrase_field(phrase, "text") or ""
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                times = "[no timing]"
            else:
                times = f"[{start:7.3f} → {end:7.3f}]"
            item = QListWidgetItem(f"{times}  {text}")
            # store phrase index for highlighting
            item.setData(Qt.ItemDataRole.UserRole, idx)
            # left-aligned text for the phrases column
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            self.kara_scroll_list.addItem(item)

    def _refresh_karaoke_words(self, line_index: Optional[int]) -> None:
        """
        Populate the words column with all words belonging to the given lyrics line_index.
        If line_index is None, the list is cleared.
        """
        if not hasattr(self, "kara_words_list"):
            return

        self.kara_words_list.clear()

        if line_index is None or not self._words:
            return

        # Filter words for the given line_index, keeping their global index
        filtered = []
        for global_idx, w in enumerate(self._words):
            if self._get_word_field(w, "line_index") == line_index:
                filtered.append((global_idx, w))

        if not filtered:
            return

        # Sort words primarily by start time, then by word_index
        def _word_sort_key(item):
            wobj = item[1]
            ws = self._get_word_field(wobj, "start")
            wi = self._get_word_field(wobj, "word_index")
            return (ws if ws is not None else 0.0, wi if wi is not None else 0)

        filtered.sort(key=_word_sort_key)

        for global_idx, w in filtered:
            text = self._get_word_field(w, "text") or ""
            ws = self._get_word_field(w, "start")
            we = self._get_word_field(w, "end")
            if ws is None or we is None:
                times = "[no timing]"
            else:
                times = f"[{ws:7.3f} → {we:7.3f}]"
            item = QListWidgetItem(f"{times}  {text}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            # store global word index for editing
            item.setData(Qt.ItemDataRole.UserRole, global_idx)
            self.kara_words_list.addItem(item)

    def _sync_phrase_timing_panel(self, idx: Optional[int], phrase: Any) -> None:
        """
        Update the 'Selected phrase timing' sliders and labels based on the given phrase.
        Does NOT seek audio or change selection; pure UI sync.
        Slider range is constrained by previous/next phrase to avoid overlaps.
        """
        if idx is None or idx < 0 or idx >= len(self._phrases):
            return

        raw_start = self._get_phrase_field(phrase, "start") or 0.0
        raw_end = self._get_phrase_field(phrase, "end") or raw_start

        min_s, max_s = self._get_phrase_neighbor_bounds(idx)

        # Clamp phrase inside neighbor bounds
        start = max(min_s, min(float(raw_start), max_s))
        end = max(start + 0.010, min(float(raw_end), max_s))

        min_ms = int(min_s * 1000.0)
        max_ms = int(max_s * 1000.0)

        self.phrase_start_slider.blockSignals(True)
        self.phrase_end_slider.blockSignals(True)

        self.phrase_start_slider.setRange(min_ms, max_ms)
        self.phrase_end_slider.setRange(min_ms, max_ms)

        self.phrase_start_slider.setValue(int(start * 1000.0))
        self.phrase_end_slider.setValue(int(end * 1000.0))

        self.phrase_start_slider.blockSignals(False)
        self.phrase_end_slider.blockSignals(False)

        self.lbl_phrase_start_value.setText(f"Start: {start:7.3f} s")
        self.lbl_phrase_end_value.setText(f"End:   {end:7.3f} s")

    def _on_kara_phrase_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """When a phrase is selected in the karaoke list, sync sliders, seek audio, and show words."""
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        phrase = self._phrases[idx]
        start = self._get_phrase_field(phrase, "start") or 0.0
        end = self._get_phrase_field(phrase, "end") or 0.0

        # Determine line_index for words
        line_index = self._get_phrase_field(phrase, "line_index")
        if line_index is None:
            line_index = idx

        # Sync the 'Selected phrase timing' panel UI (no audio seek here)
        self._sync_phrase_timing_panel(idx, phrase)

        # Seek audio to the beginning of this phrase
        position_ms = int(start * 1000.0)
        if hasattr(self, "kara_slider"):
            if getattr(self, "_media_duration_ms", 0) > 0 and self.kara_slider.maximum() == 0:
                self.kara_slider.setRange(0, self._media_duration_ms)

            self.kara_slider.blockSignals(True)
            self.kara_slider.setValue(position_ms)
            self.kara_slider.blockSignals(False)

        if hasattr(self, "lbl_kara_track_time"):
            self.lbl_kara_track_time.setText(f"{start:7.3f} s")

        self.player.setPosition(position_ms)

        # Update karaoke labels/highlight immediately
        self._update_karaoke_for_time(start)

        # Update the words column for this phrase
        self._refresh_karaoke_words(line_index)


    def _on_phrase_start_slider_moved(self, position_ms: int) -> None:
        """Update start label while dragging the start slider."""
        seconds = position_ms / 1000.0
        self.lbl_phrase_start_value.setText(f"Start: {seconds:7.3f} s")

    def _on_phrase_end_slider_moved(self, position_ms: int) -> None:
        """Update end label while dragging the end slider."""
        seconds = position_ms / 1000.0
        self.lbl_phrase_end_value.setText(f"End:   {seconds:7.3f} s")

    def _on_phrase_sliders_released(self) -> None:
        """
        When either start/end slider is released, apply the new timings
        to the selected phrase, clamped between previous/next phrases
        so that no overlap is created, then save to disk.
        """
        current = self.kara_scroll_list.currentItem()
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        min_s, max_s = self._get_phrase_neighbor_bounds(idx)

        start_s = self.phrase_start_slider.value() / 1000.0
        end_s = self.phrase_end_slider.value() / 1000.0

        # Clamp to neighbor bounds
        if start_s < min_s:
            start_s = min_s
        if end_s > max_s:
            end_s = max_s

        # Ensure end > start
        if end_s <= start_s:
            end_s = min(start_s + 0.010, max_s)
            if end_s <= start_s:
                # Degenerate case: force minimal span inside [min_s, max_s]
                start_s = max(min_s, max_s - 0.010)
                end_s = max_s

        # Push clamped values back to sliders
        self.phrase_start_slider.blockSignals(True)
        self.phrase_end_slider.blockSignals(True)
        self.phrase_start_slider.setValue(int(start_s * 1000.0))
        self.phrase_end_slider.setValue(int(end_s * 1000.0))
        self.phrase_start_slider.blockSignals(False)
        self.phrase_end_slider.blockSignals(False)

        phrase = self._phrases[idx]
        self._set_phrase_field(phrase, "start", float(start_s))
        self._set_phrase_field(phrase, "end", float(end_s))

        # Refresh both alignment list and karaoke list
        self.populate_segments_preview()

        # Restore selection in both lists
        if 0 <= idx < self.segments_list.count():
            self.segments_list.setCurrentRow(idx)
        if 0 <= idx < self.kara_scroll_list.count():
            self.kara_scroll_list.setCurrentRow(idx)

        # Update status and labels
        self.lbl_align_status.setText(f"Updated phrase {idx + 1} from sliders.")
        self.lbl_phrase_start_value.setText(f"Start: {start_s:7.3f} s")
        self.lbl_phrase_end_value.setText(f"End:   {end_s:7.3f} s")

        # Persist new timings to JSON + SRT
        self.save_timings_to_files()

    def on_segment_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        phrase = self._phrases[idx]
        start = self._get_phrase_field(phrase, "start") or 0.0
        end = self._get_phrase_field(phrase, "end") or 0.0

        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)
        self.spin_start.setValue(float(start))
        self.spin_end.setValue(float(end))
        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)

    def apply_timing_to_selected(self):
        current = self.segments_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No phrase selected", "Select a phrase in the list first.")
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if idx is None or not isinstance(idx, int):
            return
        if idx < 0 or idx >= len(self._phrases):
            return

        start = float(self.spin_start.value())
        end = float(self.spin_end.value())
        if end <= start:
            QMessageBox.warning(self, "Invalid timing", "End time must be greater than start time.")
            return

        phrase = self._phrases[idx]
        self._set_phrase_field(phrase, "start", start)
        self._set_phrase_field(phrase, "end", end)

        text = self._get_phrase_field(phrase, "text") or ""
        times = f"[{start:7.3f} → {end:7.3f}]"
        current.setText(f"{idx + 1:03d} {times}  {text}")
        self.lbl_align_status.setText(f"Updated timing of phrase {idx + 1}.")

    # ------------------------------------------------------------------
    # Save timings → JSON + SRT
    # ------------------------------------------------------------------

    def _format_srt_timestamp(self, t: float) -> str:
        if t < 0:
            t = 0.0
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        millis = int(round((t - int(t)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    def _parse_srt_timestamp(self, s: str) -> float:
        """Parse 'HH:MM:SS,mmm' or 'HH:MM:SS.mmm' into float seconds."""
        s = s.strip().replace(",", ".")
        parts = s.split(":")
        if len(parts) != 3:
            return 0.0
        h, m, sec = parts
        try:
            return int(h) * 3600 + int(m) * 60 + float(sec)
        except ValueError:
            return 0.0

    def save_timings_to_files(self):
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        if not self._phrases:
            QMessageBox.warning(self, "No phrases", "No alignment has been run yet.")
            return

        align_dir = self._project.folder / "vocal_align"
        align_dir.mkdir(parents=True, exist_ok=True)
        phrases_path = align_dir / "phrases.json"
        srt_path = align_dir / "subtitles.srt"
        words_path = align_dir / "words.json"

        # Save phrases JSON (convert to plain dicts so dataclasses are serializable)
        try:
            phrases_json = []
            for phrase in self._phrases:
                phrases_json.append(
                    {
                        "line_index": self._get_phrase_field(phrase, "line_index"),
                        "text": self._get_phrase_field(phrase, "text"),
                        "start": self._get_phrase_field(phrase, "start"),
                        "end": self._get_phrase_field(phrase, "end"),
                    }
                )

            with phrases_path.open("w", encoding="utf-8") as f:
                json.dump(phrases_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write phrases.json:\n{e}")
            return

        # Save words JSON (if any) – déjà en dicts uniformisés
        try:
            words_json = []
            for w in self._words:
                words_json.append(
                    {
                        "line_index": self._get_word_field(w, "line_index"),
                        "word_index": self._get_word_field(w, "word_index"),
                        "text": self._get_word_field(w, "text"),
                        "matched_text": self._get_word_field(w, "matched_text"),
                        "start": self._get_word_field(w, "start"),
                        "end": self._get_word_field(w, "end"),
                    }
                )
            with words_path.open("w", encoding="utf-8") as f:
                json.dump(words_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write words.json:\n{e}")
            return

        # Save SRT (comme avant)
        try:
            with srt_path.open("w", encoding="utf-8") as f:
                idx_srt = 1
                for phrase in self._phrases:
                    text = self._get_phrase_field(phrase, "text") or ""
                    start = self._get_phrase_field(phrase, "start")
                    end = self._get_phrase_field(phrase, "end")
                    if not text or start is None or end is None:
                        continue
                    start_srt = self._format_srt_timestamp(float(start))
                    end_srt = self._format_srt_timestamp(float(end))
                    f.write(f"{idx_srt}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(text + "\n\n")
                    idx_srt += 1
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not write SRT file:\n{e}")
            return

        # Status message only (no popup)
        self.lbl_align_status.setText(
            f"Timings saved to phrases.json, words.json and SRT."
        )

    def _get_project_id_for_export(self) -> str:
        """Return a short ID used to suffix exported filenames."""
        if self._project is not None and getattr(self._project, "id", None):
            return self._project.id
        return "project"

    def _get_alignment_dir(self) -> Optional[Path]:
        """Return the vocal_align folder for the current project."""
        if not self._project:
            return None
        return self._project.folder / "vocal_align"

    # -------------------- SRT --------------------

    def export_srt(self) -> None:
        """Export subtitles.srt (for YouTube / DaVinci Resolve)."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        align_dir = self._get_alignment_dir()
        assert align_dir is not None
        src = align_dir / "subtitles.srt"
        if not src.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                "No 'subtitles.srt' found in the vocal_align folder.\n"
                "Please run the alignment and save timings first.",
            )
            return

        project_id = self._get_project_id_for_export()
        default_name = f"subtitles_{project_id}.srt"
        default_path = align_dir / default_name

        dest_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export SRT subtitles",
            str(default_path),
            "SubRip subtitles (*.srt);;All files (*.*)",
        )
        if not dest_path_str:
            return

        dest_path = Path(dest_path_str)
        try:
            shutil.copy2(src, dest_path)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not export SRT file:\n{e}")
            return

        self.lbl_export_status.setText(f"Exported SRT to: {dest_path}")

    # -------------------- JSON / CSV helpers --------------------

    def _export_json_with_dialog(self, src_name: str, default_base: str, title: str) -> None:
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return
        align_dir = self._get_alignment_dir()
        assert align_dir is not None
        src = align_dir / src_name
        if not src.is_file():
            QMessageBox.warning(
                self,
                "File not found",
                f"No '{src_name}' found in the vocal_align folder.\n"
                "Please run the alignment and save timings first.",
            )
            return
        project_id = self._get_project_id_for_export()
        default_name = f"{default_base}_{project_id}.json"
        default_path = align_dir / default_name

        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            title,
            str(default_path),
            "JSON files (*.json);;All files (*.*)",
        )
        if not dest_str:
            return
        dest = Path(dest_str)
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not export JSON file:\n{e}")
            return
        self.lbl_export_status.setText(f"Exported {src_name} to: {dest}")


    # -------------------- Phrases --------------------

    def export_phrases_json(self) -> None:
        self._export_json_with_dialog(
            src_name="phrases.json",
            default_base="phrases",
            title="Export phrase timings (JSON)",
        )

    # -------------------- Words --------------------

    def export_words_json(self) -> None:
        self._export_json_with_dialog(
            src_name="words.json",
            default_base="words",
            title="Export word timings (JSON)",
        )

    # -------------------- Phonemes --------------------

    def export_phonemes_json(self) -> None:
        self._export_json_with_dialog(
            src_name="phonemes.json",
            default_base="phonemes",
            title="Export phoneme timings (JSON)",
        )

    # -------------------- Lyrics TXT --------------------

    def export_lyrics_txt(self) -> None:
        """Export lyrics text as a plain TXT file."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        text = getattr(self._project, "lyrics_text", "") or ""
        if not text.strip():
            QMessageBox.warning(
                self,
                "No lyrics",
                "No lyrics text is stored for this project.",
            )
            return

        project_id = self._get_project_id_for_export()
        default_name = f"lyrics_{project_id}.txt"
        default_path = self._project.folder / default_name

        dest_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export lyrics (TXT)",
            str(default_path),
            "Text files (*.txt);;All files (*.*)",
        )
        if not dest_str:
            return
        dest = Path(dest_str)

        try:
            dest.write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Export error", f"Could not write TXT file:\n{e}")
            return

        self.lbl_export_status.setText(f"Exported lyrics TXT to: {dest}")


    def export_alignment_files(self) -> None:
        """Export SRT, words/phonemes JSON and lyrics TXT to a user-chosen folder."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        align_dir = self._project.folder / "vocal_align"
        if not align_dir.is_dir():
            QMessageBox.warning(
                self,
                "No alignment folder",
                "No 'vocal_align' folder was found for this project.\n"
                "Please run the alignment first.",
            )
            return

        # Let the user choose an export directory (e.g. for YouTube / DaVinci)
        target_dir_str = QFileDialog.getExistingDirectory(
            self,
            "Choose export folder",
            str(self._project.folder),
        )
        if not target_dir_str:
            # User cancelled
            return

        target_dir = Path(target_dir_str)

        # Files we try to export. They are produced by the alignment pipeline.
        files_to_copy = [
            "subtitles.srt",   # phrases to SRT (YouTube / DaVinci Resolve)
            "phrases.json",    # phrase-level timings
            "words.json",      # word-level timings
            "phonemes.json",   # phoneme-level timings (if created)
            "lyrics.txt",      # plain lyrics text
        ]

        copied = 0
        missing = []

        for name in files_to_copy:
            src = align_dir / name
            if src.is_file():
                try:
                    shutil.copy2(src, target_dir / name)
                    copied += 1
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Export error",
                        f"Could not copy '{name}':\n{e}",
                    )
            else:
                missing.append(name)

        msg = f"Exported {copied} file(s) to:\n{target_dir}"
        if missing:
            msg += "\nMissing (not created yet): " + ", ".join(missing)

        self.lbl_export_status.setText(msg)

    # ==================== IMPORT FUNCTIONS ====================

    def import_phrases_json(self) -> None:
        """Import phrase timings from a JSON file and replace current phrases."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import phrase timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of phrase objects.",
            )
            return

        self._phrases = data
        self.populate_segments_preview()
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported phrase timings from: {path}")

    def import_words_json(self) -> None:
        """Import word timings from a JSON file and replace current words."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import word timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of word objects.",
            )
            return

        self._words = data
        self._refresh_karaoke_words(None)
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported word timings from: {path}")

    def import_phonemes_json(self) -> None:
        """Import phoneme timings JSON and overwrite phonemes.json in vocal_align."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import phoneme timings (JSON)",
            "JSON files (*.json);;All files (*.*)",
        )
        if path is None:
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read JSON file:\n{e}")
            return

        if not isinstance(data, list):
            QMessageBox.critical(
                self,
                "Import error",
                "Expected a JSON list of phoneme objects.",
            )
            return

        align_dir = self._get_alignment_dir()
        target = align_dir / "phonemes.json"
        try:
            target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not write phonemes.json:\n{e}")
            return

        self.lbl_export_status.setText(f"Imported phoneme timings from: {path}")

    def import_lyrics_txt(self) -> None:
        """Import lyrics from a TXT file and replace the current lyrics."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        path = self._open_file_dialog(
            "Import lyrics (TXT)",
            "Text files (*.txt);;All files (*.*)",
        )
        if path is None:
            return

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read TXT file:\n{e}")
            return

        # Update project object and text editor
        self.lyrics_edit.setPlainText(text)
        if hasattr(self._project, "lyrics_text"):
            self._project.lyrics_text = text

        # Also write into project folder and vocal_align folder
        try:
            (self._project.folder / "lyrics.txt").write_text(text, encoding="utf-8")
            align_dir = self._get_alignment_dir()
            (align_dir / "lyrics.txt").write_text(text, encoding="utf-8")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not write lyrics.txt:\n{e}")
            return

        self.lbl_export_status.setText(f"Imported lyrics TXT from: {path}")


    def import_srt(self) -> None:
        """Import an SRT file and replace phrase timings accordingly."""
        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        src_path = self._open_file_dialog(
            "Import SRT subtitles",
            "SubRip subtitles (*.srt);;All files (*.*)",
        )
        if src_path is None:
            return

        try:
            content = src_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            QMessageBox.critical(self, "Import error", f"Could not read SRT file:\n{e}")
            return

        # Split into SRT blocks without using regex
        blocks: List[str] = []
        current: List[str] = []
        for line in content.splitlines():
            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current))
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current))

        new_phrases: List[dict] = []
        for block in blocks:
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            if len(lines) < 2:
                continue

            # Optional numeric index on first line
            idx_line = 0
            if lines[0].isdigit():
                idx_line = 1
            if idx_line >= len(lines):
                continue

            timing_line = lines[idx_line]
            if "-->" not in timing_line:
                continue
            start_raw, end_raw = timing_line.split("-->", 1)
            start = self._parse_srt_timestamp(start_raw.strip())
            end = self._parse_srt_timestamp(end_raw.strip())
            if end <= start:
                continue

            text = " ".join(lines[idx_line + 1 :]) if len(lines) > idx_line + 1 else ""
            new_phrases.append(
                {
                    "line_index": len(new_phrases),
                    "text": text,
                    "start": float(start),
                    "end": float(end),
                }
            )

        if not new_phrases:
            QMessageBox.warning(
                self,
                "Import error",
                "No valid entries were found in the SRT file.",
            )
            return

        # Replace in-memory phrases
        self._phrases = new_phrases

        # Refresh UI (lists + karaoke preview)
        self.populate_segments_preview()

        # Persist everything (phrases.json, words.json, phonemes.json, subtitles.srt)
        self.save_timings_to_files()
        self.lbl_export_status.setText(f"Imported SRT from: {src_path}")


    # ------------------------------------------------------------------
    # Karaoke playback controls
    # ------------------------------------------------------------------

    def play_karaoke(self):
        """Play or resume the selected audio (full mix or manual file)."""

        if not self._project:
            QMessageBox.warning(self, "No project", "Please select a project first.")
            return

        audio_path = self._get_audio_path_for_current_mode()
        if not audio_path or not Path(audio_path).is_file():
            mode = "full_mix"
            if hasattr(self, "audio_source_combo") and self.audio_source_combo is not None:
                data = self.audio_source_combo.currentData()
                if isinstance(data, str) and data:
                    mode = data

            if mode == "manual":
                QMessageBox.warning(
                    self,
                    "Invalid audio file",
                    "No valid manual audio file selected.\n"
                    "Please click 'Browse…' or switch the audio source to 'Full mix (project audio)'.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "No audio file",
                    "This project has no valid audio file associated.",
                )
            return

        url = QUrl.fromLocalFile(str(audio_path.resolve()))

        # If the current source differs from the desired audio, load it and start from 0
        if self.player.source() != url:
            self.player.setSource(url)
            self.player.setPosition(0)
        else:
            # If we are in StoppedState, ensure we start from 0
            if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
                self.player.setPosition(0)
            # If we are in PausedState, keep current position and just resume

        # Start or resume playback
        self.player.play()

        # Notify global player bar if connected
        try:
            mode = self.audio_source_combo.currentData() if self.audio_source_combo is not None else "full_mix"
            label_mode = "manual file" if mode == "manual" else "full mix"
            self.audioStarted.emit(f"Karaoke: {self._project.name} — {label_mode}")
        except Exception:
            pass


    def stop_karaoke(self):
        """Stop playback for karaoke."""
        self.player.stop()

    def pause_karaoke(self):
        """Pause playback for karaoke."""
        self.player.pause()

    def _on_kara_slider_moved(self, position_ms: int) -> None:
        """Seek in the current audio when moving the karaoke slider and update karaoke preview."""
        # Seek the shared player
        self.player.setPosition(position_ms)

        # Update local UI immediately (even before positionChanged is emitted)
        if hasattr(self, "lbl_kara_track_time"):
            seconds = position_ms / 1000.0
            self.lbl_kara_track_time.setText(f"{seconds:7.3f} s")

        if self._phrases:
            pos_s = position_ms / 1000.0
            self._update_karaoke_for_time(pos_s)



    def _on_player_position_changed_karaoke(self, position_ms: int) -> None:
        """Update slider and time label for karaoke playback."""
        if not hasattr(self, "kara_position_slider"):
            return
        self.kara_position_slider.blockSignals(True)
        self.kara_position_slider.setValue(position_ms)
        self.kara_position_slider.blockSignals(False)
        seconds = position_ms / 1000.0
        if hasattr(self, "lbl_kara_time"):
            self.lbl_kara_time.setText(f"{seconds:7.3f} s")

    def _on_player_duration_changed_karaoke(self, duration_ms: int) -> None:
        """Update slider range when media duration changes."""
        if not hasattr(self, "kara_position_slider"):
            return
        self.kara_position_slider.setRange(0, duration_ms)

    def _on_karaoke_slider_moved(self, position_ms: int) -> None:
        """Seek in the current audio when moving the karaoke slider."""
        self.player.setPosition(position_ms)

    def apply_time_offsets(self) -> None:
        """Apply global offsets to phrase and word timings."""
        phrase_offset = float(self.spin_phrase_offset.value()) if hasattr(self, "spin_phrase_offset") else 0.0
        word_offset = float(self.spin_word_offset.value()) if hasattr(self, "spin_word_offset") else 0.0

        if not self._phrases and not self._words:
            QMessageBox.warning(self, "No alignment", "No alignment has been run yet.")
            return

        if phrase_offset == 0.0 and word_offset == 0.0:
            QMessageBox.information(self, "No change", "Offsets are both zero; nothing to apply.")
            return

        # Apply phrase offset to all phrases
        if phrase_offset != 0.0:
            for phrase in self._phrases:
                start = self._get_phrase_field(phrase, "start")
                end = self._get_phrase_field(phrase, "end")
                if start is not None:
                    self._set_phrase_field(phrase, "start", float(start) + phrase_offset)
                if end is not None:
                    self._set_phrase_field(phrase, "end", float(end) + phrase_offset)

        # Apply word offset to all words
        if word_offset != 0.0:
            for w in self._words:
                ws = self._get_word_field(w, "start")
                we = self._get_word_field(w, "end")
                if ws is not None:
                    if isinstance(w, dict):
                        w["start"] = float(ws) + word_offset
                    else:
                        setattr(w, "start", float(ws) + word_offset)
                if we is not None:
                    if isinstance(w, dict):
                        w["end"] = float(we) + word_offset
                    else:
                        setattr(w, "end", float(we) + word_offset)

        # Refresh views and status
        self.populate_segments_preview()
        self.lbl_align_status.setText("Applied phrase/word offsets.")




    # ------------------------------------------------------------------
    # Karaoke preview helper
    # ------------------------------------------------------------------
    
    
    def _update_karaoke_scroll_highlight(self, current_idx: Optional[int]) -> None:
        """
        Update colors and bold style in the karaoke scrolling list:
        previous / current / next.

        Default text color is taken from the current palette so the list
        stays readable across light / dark / neon themes.
        """
        if not hasattr(self, "kara_scroll_list"):
            return

        # Use the QListWidget's palette as the base text color
        palette = self.kara_scroll_list.palette()
        default_color = palette.color(QPalette.ColorRole.Text)

        for row in range(self.kara_scroll_list.count()):
            item = self.kara_scroll_list.item(row)
            idx = item.data(Qt.ItemDataRole.UserRole)

            # Default style: normal weight + theme text color
            color = default_color
            font = item.font()
            font.setBold(False)

            if isinstance(idx, int) and current_idx is not None:
                if idx == current_idx - 1:
                    # Previous phrase: red
                    color = QColor(200, 0, 0)
                elif idx == current_idx:
                    # Current phrase: green + bold
                    color = QColor(0, 150, 0)
                    font.setBold(True)
                elif idx == current_idx + 1:
                    # Next phrase: orange
                    color = QColor(220, 120, 0)

            item.setForeground(color)
            item.setFont(font)

            # Keep current phrase visible in the viewport
            if isinstance(idx, int) and current_idx is not None and idx == current_idx:
                self.kara_scroll_list.scrollToItem(item)

    def _update_karaoke_word_highlight(
        self,
        line_index: Optional[int],
        global_word_idx: Optional[int],
    ) -> None:
        """
        Highlight the current word in the right words list:
        - bold + blue for the current word
        - normal + theme text color for others.
        """
        if not hasattr(self, "kara_words_list"):
            return

        # Use the words list palette so colors remain consistent with the theme
        palette = self.kara_words_list.palette()
        default_color = palette.color(QPalette.ColorRole.Text)

        for row in range(self.kara_words_list.count()):
            item = self.kara_words_list.item(row)
            gidx = item.data(Qt.ItemDataRole.UserRole)

            font = item.font()
            if global_word_idx is not None and isinstance(gidx, int) and gidx == global_word_idx:
                font.setBold(True)
                item.setFont(font)
                # Current word: blue highlight
                item.setForeground(QColor(0, 0, 220))
                # Keep highlighted word visible
                self.kara_words_list.scrollToItem(item)
            else:
                font.setBold(False)
                item.setFont(font)
                # Other words: normal theme text color
                item.setForeground(default_color)

    def _update_karaoke_for_time(self, pos_s: float):
        """
        Set current line + word labels and highlights based on current playback time.
        Also keeps the words column synchronized with the currently sung phrase.
        """
        current_phrase = None
        current_idx: Optional[int] = None
        phrase_start = None
        phrase_end = None

        # Find current phrase for the given time
        for idx, phrase in enumerate(self._phrases):
            start = self._get_phrase_field(phrase, "start")
            end = self._get_phrase_field(phrase, "end")
            if start is None or end is None:
                continue
            if start <= pos_s <= end:
                current_phrase = phrase
                current_idx = idx
                phrase_start = start
                phrase_end = end
                break

        if current_phrase is None:
            # Outside any phrase: hide karaoke line/word, clear highlight and words list
            self.lbl_kara_line.setText("")
            self.lbl_kara_word.setText("")
            self._update_karaoke_scroll_highlight(None)
            self._refresh_karaoke_words(None)
            # Clear word highlight state
            if hasattr(self, "_kara_current_line_index"):
                self._kara_current_line_index = None
            self._update_karaoke_word_highlight(None, None)
            return

        # Update scrolling lyrics highlight (previous / current / next)
        self._update_karaoke_scroll_highlight(current_idx)

        # Phrase text + timer
        line_text = self._get_phrase_field(current_phrase, "text") or ""
        if phrase_start is not None and phrase_end is not None:
            self.lbl_kara_line.setText(
                f"[{phrase_start:7.3f} → {phrase_end:7.3f}]  {line_text}"
            )
        else:
            self.lbl_kara_line.setText(line_text)

        # Determine which line_index is used for words
        line_index = self._get_phrase_field(current_phrase, "line_index")
        if line_index is None:
            line_index = current_idx

        # Make sure the words list shows the words for the current phrase
        if not hasattr(self, "_kara_current_line_index"):
            self._kara_current_line_index = None
        if self._kara_current_line_index != line_index:
            self._refresh_karaoke_words(line_index)
            self._kara_current_line_index = line_index

        # Find current word within that line
        current_word_text = ""
        current_word_start: Optional[float] = None
        current_word_end: Optional[float] = None
        current_word_global_idx: Optional[int] = None

        for gidx, w in enumerate(self._words or []):
            if self._get_word_field(w, "line_index") != line_index:
                continue
            ws = self._get_word_field(w, "start")
            we = self._get_word_field(w, "end")
            if ws is None or we is None:
                continue
            if ws <= pos_s <= we:
                current_word_text = self._get_word_field(w, "text") or ""
                current_word_start = ws
                current_word_end = we
                current_word_global_idx = gidx
                break

        # Update word label with timer
        if current_word_text:
            if current_word_start is not None and current_word_end is not None:
                self.lbl_kara_word.setText(
                    f"[{current_word_start:7.3f} → {current_word_end:7.3f}]  {current_word_text}"
                )
            else:
                self.lbl_kara_word.setText(current_word_text)
        else:
            self.lbl_kara_word.setText("")

        # Highlight current word in the right column
        self._update_karaoke_word_highlight(line_index, current_word_global_idx)
