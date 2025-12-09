from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QFontMetrics,
    QLinearGradient,
    QBrush,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from olaf_app.lyrics_visualization_api import BaseLyricsVisualization, LyricsFrameContext
from olaf_app.visualization_api import PluginParameter
from olaf_app.lyrics_text_style import (
    apply_default_text_style_config,
    text_style_parameters,
    build_qfont_from_config,
    font_color_from_config,
    draw_styled_text,
)


class VerticalScrollLyricsVisualization(BaseLyricsVisualization):
    """
    Vertical word-by-word scroll inside a configurable lyrics box.

    Design intent:
      - Visual dédié aux chansons à tempo lent / moyen, avec des paroles
        plutôt espacées (ballades, doom, ambient, etc.).
      - Chaque phrase est affichée mot par mot (un mot par ligne) dans
        une zone rectangulaire dédiée ("lyrics box") persistante.
      - La phrase apparaît en bas de cette zone, puis se déplace vers le haut.
      - Les mots disparaissent simplement lorsqu'ils sortent par le haut
        de la lyrics box (pas de fade-out mot par mot).

    Fading (niveau phrase):
      - Toute la phrase a un fade-in temporel au début
        (phrase_fade_in_time).
      - Toute la phrase a un fade-out temporel pendant les dernières
        phrase_fade_out_time secondes de la phrase, avec une valeur
        maximale de 0.5 s (donc la phrase disparaît au plus 0.5 s avant
        la fin de la phrase / l'apparition de la suivante).
      - Le highlight du mot actif reste possible, mais sa luminosité
        est modulée par l'opacité de la phrase.

    Important:
      - La "background box" (text_box_*) reste un pur décor de texte
        partagé par tous les plugins (via draw_styled_text).
      - La zone de défilement (lyrics box) est entièrement gérée par
        ce plugin via les paramètres lyrics_box_* et ne dépend PAS de
        text_box_padding.
    """

    plugin_id: str = "lyrics_vertical_scroll"
    plugin_name: str = "Vertical word-by-word scroll (slow tempo required)"
    plugin_description: str = (
        "Scrolls each phrase vertically inside a configurable lyrics box, "
        "one word per line. The whole phrase fades in at the beginning "
        "and fades out near the end (up to 0.5 s before the next phrase), "
        "while words simply leave the frame at the top."
    )
    plugin_author: str = "Olaf"
    plugin_version: str = "1.4.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Ensure shared text-style defaults are present in the config.
        apply_default_text_style_config(self.config)

        # Backward compatibility: migrate old box_* keys to lyrics_box_*
        if "lyrics_box_left_margin" not in self.config and "box_left_margin" in self.config:
            self.config["lyrics_box_left_margin"] = self.config.get("box_left_margin", 0.15)
        if "lyrics_box_right_margin" not in self.config and "box_right_margin" in self.config:
            self.config["lyrics_box_right_margin"] = self.config.get("box_right_margin", 0.15)
        if "lyrics_box_top_margin" not in self.config and "box_top_margin" in self.config:
            self.config["lyrics_box_top_margin"] = self.config.get("box_top_margin", 0.20)
        if "lyrics_box_bottom_margin" not in self.config and "box_bottom_margin" in self.config:
            self.config["lyrics_box_bottom_margin"] = self.config.get("box_bottom_margin", 0.15)

        # Smoothed amplitude for background / highlight reactivity.
        self._smoothed_amp: float = 0.0

        # Cache of phrase layout data, keyed by phrase_index.
        #   {
        #       "text": str,
        #       "words": List[Dict[str, Any]],  # {"text", "char_start", "char_end"}
        #   }
        self._phrase_layout: Dict[int, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # Plugin-specific defaults (lyrics box + scroll + fades)
        # ------------------------------------------------------------------

        # Geometry of the lyrics display box, expressed as margins relative to
        # the widget size. This is the scroll area for the words.
        self.config.setdefault("lyrics_box_left_margin", 0.15)
        self.config.setdefault("lyrics_box_right_margin", 0.15)
        self.config.setdefault("lyrics_box_top_margin", 0.0)
        self.config.setdefault("lyrics_box_bottom_margin", 0.20)

        # Vertical spacing between consecutive word baselines (in multiples
        # of font height).
        self.config.setdefault("line_spacing_factor", 1.0)

        # Additional time (in seconds) before the last word of a phrase
        # reaches the top of the lyrics box: ralentit le défilement global
        # et permet un léger overlap visuel entre phrases.
        self.config.setdefault("extra_travel_time", 3.5)

        # Fade-in temporel (seconds) au début de la phrase.
        self.config.setdefault("phrase_fade_in_time", 1.0)

        # Fade-out temporel (seconds) sur la fin de la phrase.
        # Limité à 0.5 s pour rester "au plus 0.5 s avant la suivante".
        self.config.setdefault("phrase_fade_out_time", 0.25)

        # Highlight geometry + color for the active word.
        self.config.setdefault("highlight_padding_x", 0.80)
        self.config.setdefault("highlight_padding_y", 0.0)
        self.config.setdefault("highlight_color", "#ffff00")
        # Opacity multiplier for the highlight box (0.0–1.0).
        self.config.setdefault("highlight_alpha", 0.8)

        # Reasonable widget size.
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Parameters metadata
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose shared text-style parameters + plugin-specific controls.
        """
        params = text_style_parameters()
        params.update(
            {
                # Lyrics box geometry (relative margins)
                "lyrics_box_left_margin": PluginParameter(
                    name="lyrics_box_left_margin",
                    label="Lyrics box left margin",
                    type="float",
                    default=0.15,
                    minimum=0.0,
                    maximum=0.45,
                    step=0.01,
                    description="Left margin of the lyrics box (fraction of width).",
                ),
                "lyrics_box_right_margin": PluginParameter(
                    name="lyrics_box_right_margin",
                    label="Lyrics box right margin",
                    type="float",
                    default=0.15,
                    minimum=0.0,
                    maximum=0.45,
                    step=0.01,
                    description="Right margin of the lyrics box (fraction of width).",
                ),
                "lyrics_box_top_margin": PluginParameter(
                    name="lyrics_box_top_margin",
                    label="Lyrics box top margin",
                    type="float",
                    default=0.20,
                    minimum=0.0,
                    maximum=0.45,
                    step=0.01,
                    description="Top margin of the lyrics box (fraction of height).",
                ),
                "lyrics_box_bottom_margin": PluginParameter(
                    name="lyrics_box_bottom_margin",
                    label="Lyrics box bottom margin",
                    type="float",
                    default=0.15,
                    minimum=0.0,
                    maximum=0.45,
                    step=0.01,
                    description="Bottom margin of the lyrics box (fraction of height).",
                ),

                # Phrase vertical spacing
                "line_spacing_factor": PluginParameter(
                    name="line_spacing_factor",
                    label="Line spacing factor",
                    type="float",
                    default=1.20,
                    minimum=0.5,
                    maximum=4.0,
                    step=0.05,
                    description=(
                        "Vertical space between consecutive words "
                        "(in multiples of font height)."
                    ),
                ),

                # Timing (global scroll + post-phrase hold)
                "extra_travel_time": PluginParameter(
                    name="extra_travel_time",
                    label="Post-phrase hold (s)",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    description=(
                        "Time the phrase stays on screen after its last "
                        "word has been sung, before the last word reaches "
                        "the top of the lyrics box."
                    ),
                ),

                # Phrase-level fades
                "phrase_fade_in_time": PluginParameter(
                    name="phrase_fade_in_time",
                    label="Phrase fade-in (s)",
                    type="float",
                    default=0.3,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description=(
                        "Duration of the fade-in for the whole phrase "
                        "at the beginning."
                    ),
                ),
                "phrase_fade_out_time": PluginParameter(
                    name="phrase_fade_out_time",
                    label="Phrase fade-out (s)",
                    type="float",
                    default=0.4,
                    minimum=0.0,
                    maximum=0.5,
                    step=0.05,
                    description=(
                        "Duration of the fade-out for the whole phrase "
                        "at the end. Capped at 0.5 s so the phrase "
                        "disappears at most 0.5 s before the next one."
                    ),
                ),

                # Highlight
                "highlight_padding_x": PluginParameter(
                    name="highlight_padding_x",
                    label="Highlight padding X (rel.)",
                    type="float",
                    default=0.40,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description=(
                        "Horizontal padding around the active word "
                        "(relative to text height)."
                    ),
                ),
                "highlight_padding_y": PluginParameter(
                    name="highlight_padding_y",
                    label="Highlight padding Y (rel.)",
                    type="float",
                    default=0.30,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description=(
                        "Vertical padding around the active word "
                        "(relative to text height)."
                    ),
                ),
                "highlight_color": PluginParameter(
                    name="highlight_color",
                    label="Highlight color",
                    type="color",
                    default="#ffff00",
                    description="Fill color for the active word highlight.",
                ),
                "highlight_alpha": PluginParameter(
                    name="highlight_alpha",
                    label="Highlight opacity",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity multiplier for the active word highlight box.",
                ),
            }
        )
        return params

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------
    def _compute_phrase_layout_and_active_index(
        self, ctx: LyricsFrameContext
    ) -> Tuple[Optional[int], List[Dict[str, Any]], Optional[int]]:
        """
        Helper used to:
        - resolve current phrase index,
        - get or build the phrase layout (list of words with char ranges),
        - find the active word index for this phrase, if any.
        """
        if ctx is None or ctx.phrase_index is None:
            return None, [], None

        phrase_index = int(ctx.phrase_index)
        phrase_text = ctx.text_full_line or ""
        layout = self._get_phrase_layout(phrase_index, phrase_text)
        words = layout["words"]

        if not words:
            return phrase_index, words, None

        active_index: Optional[int] = None

        # Prefer char offsets if available
        if ctx.word_char_start is not None:
            start_char = int(ctx.word_char_start)
            for i, winfo in enumerate(words):
                if winfo["char_start"] <= start_char < winfo["char_end"]:
                    active_index = i
                    break

        # Fallback: text match on the word
        if active_index is None and ctx.text_active_word:
            target = ctx.text_active_word.strip().lower()
            for i, winfo in enumerate(words):
                if str(winfo["text"]).strip().lower() == target:
                    active_index = i
                    break

        return phrase_index, words, active_index

    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        - Smooth amplitude (for highlight / background).

        Toute la logique de fade est au niveau de la phrase
        dans paintEvent (plus de tracking mot par mot).
        """
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.25
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _split_phrase_words(self, text: str) -> List[Dict[str, Any]]:
        """
        Split a phrase into "words" based on runs of non-whitespace
        characters so we can map char offsets to word indices.
        """
        words: List[Dict[str, Any]] = []
        if not text:
            return words

        start: Optional[int] = None
        for idx, ch in enumerate(text):
            if ch.isspace():
                if start is not None:
                    words.append(
                        {
                            "text": text[start:idx],
                            "char_start": start,
                            "char_end": idx,
                        }
                    )
                    start = None
            else:
                if start is None:
                    start = idx

        if start is not None:
            words.append(
                {
                    "text": text[start:len(text)],
                    "char_start": start,
                    "char_end": len(text),
                }
            )
        return words

    def _get_phrase_layout(self, phrase_index: int, text: str) -> Dict[str, Any]:
        """
        Return (and cache) the layout info for a phrase index.
        """
        layout = self._phrase_layout.get(phrase_index)
        if layout is not None and layout.get("text") == text:
            return layout

        words = self._split_phrase_words(text)
        layout = {"text": text, "words": words}
        self._phrase_layout[phrase_index] = layout
        return layout

    def _paint_background(self, painter: QPainter, rect) -> None:
        """
        Fill the global background (outside / behind the lyrics box).

        Les paramètres background_* viennent de la config partagée
        (text_style_parameters) et sont appliqués ici.
        """
        mode = str(self.config.get("background_mode", "gradient") or "gradient")

        # Try to retrieve a cover pixmap if the host provides one.
        cover_source = getattr(self, "cover_pixmap", None)
        cover_pixmap = None
        if callable(cover_source):
            cover_pixmap = cover_source()
        else:
            cover_pixmap = cover_source

        amp = self._smoothed_amp

        if mode == "cover" and cover_pixmap is not None and not cover_pixmap.isNull():
            pix = cover_pixmap
            pw = pix.width()
            ph = pix.height()
            if pw > 0 and ph > 0:
                rw = rect.width()
                rh = rect.height()
                scale = max(rw / pw, rh / ph)
                sw = int(pw * scale)
                sh = int(ph * scale)
                sx = rect.center().x() - sw // 2
                sy = rect.center().y() - sh // 2
                painter.drawPixmap(sx, sy, sw, sh, pix)
        elif mode == "solid":
            col_str = str(self.config.get("background_color", "#000000") or "#000000")
            color = QColor(col_str) if QColor.isValidColor(col_str) else QColor(0, 0, 0)
            painter.fillRect(rect, color)
        else:
            # Default: gradient
            top_str = str(self.config.get("background_gradient_top", "#101010") or "#101010")
            bottom_str = str(self.config.get("background_gradient_bottom", "#402840") or "#402840")
            top_color = QColor(top_str) if QColor.isValidColor(top_str) else QColor(16, 16, 16)
            bottom_color = QColor(bottom_str) if QColor.isValidColor(bottom_str) else QColor(64, 40, 64)

            # Slight reactivity on the bottom color
            bottom_color = QColor(
                min(255, bottom_color.red() + int(80 * amp)),
                bottom_color.green(),
                min(255, bottom_color.blue() + int(80 * amp)),
            )

            gradient = QLinearGradient(
                float(rect.left()),
                float(rect.top()),
                float(rect.left()),
                float(rect.bottom()),
            )
            gradient.setColorAt(0.0, top_color)
            gradient.setColorAt(1.0, bottom_color)
            painter.fillRect(rect, QBrush(gradient))

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # type: ignore[override]
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        widget_rect = self.rect()

        # Global background (gradient / solid / cover)
        self._paint_background(painter, widget_rect)

        # --------------------------------------------------------------
        # Compute lyrics box geometry from margins (scroll area)
        # --------------------------------------------------------------
        try:
            ml = float(
                self.config.get("lyrics_box_left_margin",
                                self.config.get("box_left_margin", 0.15))
            )
            mr = float(
                self.config.get("lyrics_box_right_margin",
                                self.config.get("box_right_margin", 0.15))
            )
            mt = float(
                self.config.get("lyrics_box_top_margin",
                                self.config.get("box_top_margin", 0.20))
            )
            mb = float(
                self.config.get("lyrics_box_bottom_margin",
                                self.config.get("box_bottom_margin", 0.15))
            )
        except Exception:
            ml, mr, mt, mb = 0.15, 0.15, 0.20, 0.15

        ml = max(0.0, min(0.45, ml))
        mr = max(0.0, min(0.45, mr))
        mt = max(0.0, min(0.45, mt))
        mb = max(0.0, min(0.45, mb))

        w = widget_rect.width()
        h = widget_rect.height()

        lyrics_left = widget_rect.left() + w * ml
        lyrics_right = widget_rect.right() - w * mr
        lyrics_top = widget_rect.top() + h * mt
        lyrics_bottom = widget_rect.bottom() - h * mb

        # Enforce a minimum height/width
        if lyrics_right <= lyrics_left:
            lyrics_right = lyrics_left + max(10, w * 0.1)
        if lyrics_bottom <= lyrics_top:
            lyrics_bottom = lyrics_top + max(10, h * 0.1)

        lyrics_rect = QRectF(
            lyrics_left,
            lyrics_top,
            lyrics_right - lyrics_left,
            lyrics_bottom - lyrics_top,
        )

        # Cette rect est la *seule* zone utilisée pour le scroll et
        # l'affichage du texte. Elle ne dépend pas de text_box_padding.
        text_rect = lyrics_rect

        # --------------------------------------------------------------
        # Typography: base font for all layout
        # --------------------------------------------------------------
        try:
            font_size = int(self.config.get("font_size", 40))
        except Exception:
            font_size = 40
        font_size = max(8, font_size)

        base_font = build_qfont_from_config(self.config, painter, font_size)
        painter.setFont(base_font)
        fm = QFontMetrics(base_font)

        ctx = self.current_context()
        if ctx is None or ctx.phrase_index is None:
            painter.end()
            return

        phrase_index, words, active_index = self._compute_phrase_layout_and_active_index(ctx)
        if phrase_index is None or not words:
            painter.end()
            return

        capitalize = bool(self.config.get("capitalize_all", False))

        def _formatted_text(raw: str) -> str:
            return raw.upper() if capitalize else raw

        # --------------------------------------------------------------
        # Choose font size so that all words fit horizontally in text_rect
        # --------------------------------------------------------------
        max_width_allowed = text_rect.width() * 0.92
        while font_size > 8:
            base_font = build_qfont_from_config(self.config, painter, font_size)
            painter.setFont(base_font)
            fm = QFontMetrics(base_font)

            max_text_width = 0
            for winfo in words:
                txt = _formatted_text(str(winfo["text"]))
                tw = fm.horizontalAdvance(txt)
                if tw > max_text_width:
                    max_text_width = tw

            if max_text_width <= max_width_allowed or font_size <= 9:
                break
            font_size -= 1

        base_font = build_qfont_from_config(self.config, painter, font_size)
        painter.setFont(base_font)
        fm = QFontMetrics(base_font)
        font_color = font_color_from_config(self.config)

        # --------------------------------------------------------------
        # Vertical layout & scroll
        # --------------------------------------------------------------
        try:
            line_spacing_factor = float(self.config.get("line_spacing_factor", 1.20))
        except Exception:
            line_spacing_factor = 1.20
        line_spacing_factor = max(0.5, line_spacing_factor)

        line_step = fm.height() * line_spacing_factor
        num_words = max(1, len(words))
        phrase_height = (num_words - 1) * line_step
        box_height = text_rect.height()

        # Effective duration = phrase duration + extra travel time
        try:
            extra_travel = float(self.config.get("extra_travel_time", 1.0))
        except Exception:
            extra_travel = 1.0
        extra_travel = max(0.0, extra_travel)

        phrase_duration = max(0.01, float(ctx.phrase_duration or 0.0))
        D = max(0.01, phrase_duration + extra_travel)

        local_t = float(ctx.local_phrase_time or 0.0)
        s = max(0.0, min(1.0, local_t / D))

        # Total vertical travel: from first word at bottom to last word at top
        v_pixels = box_height + phrase_height

        # Phrase-level fade-in / fade-out
        try:
            fade_in_time = float(self.config.get("phrase_fade_in_time", 0.3))
        except Exception:
            fade_in_time = 0.3
        fade_in_time = max(0.0, min(2.0, fade_in_time))

        try:
            fade_out_time = float(self.config.get("phrase_fade_out_time", 0.4))
        except Exception:
            fade_out_time = 0.4
        fade_out_time = max(0.0, min(0.5, fade_out_time))

        # Compute phrase alpha (same for all words)
        alpha_phrase = 1.0
        # Fade-in au début
        if fade_in_time > 0.0 and local_t < fade_in_time:
            alpha_phrase *= max(0.0, min(1.0, local_t / fade_in_time))
        # Fade-out sur la fin de la phrase (sans dépendre du scroll)
        if fade_out_time > 0.0 and local_t > phrase_duration - fade_out_time:
            # On ne tient pas compte du post-hold ici : fade-out collé à la fin de la phrase.
            remaining = phrase_duration - local_t
            alpha_phrase *= max(0.0, min(1.0, remaining / fade_out_time))

        amp = self._smoothed_amp

        # Highlight params
        try:
            pad_x_factor = float(self.config.get("highlight_padding_x", 0.40))
        except Exception:
            pad_x_factor = 0.40
        try:
            pad_y_factor = float(self.config.get("highlight_padding_y", 0.30))
        except Exception:
            pad_y_factor = 0.30
        pad_x_factor = max(0.0, pad_x_factor)
        pad_y_factor = max(0.0, pad_y_factor)

        col_str = str(self.config.get("highlight_color", "#ffff00") or "#ffff00")
        highlight_color = (
            QColor(col_str) if QColor.isValidColor(col_str) else QColor(255, 255, 0)
        )

        try:
            highlight_alpha_cfg = float(self.config.get("highlight_alpha", 1.0))
        except Exception:
            highlight_alpha_cfg = 1.0
        highlight_alpha_cfg = max(0.0, min(1.0, highlight_alpha_cfg))

        # --------------------------------------------------------------
        # Draw words: phrase alpha commun + disparition au bord supérieur
        # --------------------------------------------------------------
        for i, winfo in enumerate(words):
            raw_txt = str(winfo["text"])
            text_line = _formatted_text(raw_txt)

            # Baseline Y for this word:
            # baseline_i(s) = bottom - s * v_pixels + i * line_step
            baseline_y = (
                text_rect.bottom()
                - s * v_pixels
                + i * line_step
            )

            # Cull clearly off-screen words (disparition brutale en haut)
            if baseline_y < text_rect.top() - line_step:
                continue
            if baseline_y > text_rect.bottom() + line_step:
                continue

            text_width = fm.horizontalAdvance(text_line)
            x0 = text_rect.center().x() - text_width / 2.0

            final_alpha = alpha_phrase
            if final_alpha <= 0.01:
                continue

            painter.save()
            painter.setOpacity(final_alpha)

            is_active = (active_index is not None and i == active_index)

            # Highlight rectangle derrière le mot actif
            if is_active:
                padding_x = fm.height() * pad_x_factor
                padding_y = fm.height() * pad_y_factor

                bg = QColor(highlight_color)

                # Fully opaque highlight when:
                #   - phrase alpha_phrase == 1.0
                #   - highlight_alpha_cfg == 1.0
                bg_alpha = max(0.0, min(1.0, highlight_alpha_cfg * alpha_phrase))
                bg.setAlphaF(bg_alpha)

                highlight_rect = QRectF(
                    x0 - padding_x,
                    baseline_y - fm.ascent() - padding_y,
                    text_width + 2.0 * padding_x,
                    fm.height() + 2.0 * padding_y,
                )

                painter.setBrush(bg)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(highlight_rect, 8.0, 8.0)

            # Draw word with full shared styling (outline / shadow / background box)
            # Background box padding (text_box_padding) is used purely as a visual
            # decoration by draw_styled_text and does NOT affect scroll geometry.
            draw_styled_text(
                painter=painter,
                x=x0,
                y=baseline_y,
                text=text_line,
                config=self.config,
                base_font=base_font,
                base_color=font_color,
            )

            painter.restore()

        painter.end()
