from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QBrush,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from olaf_app.lyrics_visualization_api import BaseLyricsVisualization, LyricsFrameContext
from olaf_app.visualization_api import PluginParameter
from olaf_app.lyrics_text_style import (
    apply_default_text_style_config,
    text_style_parameters as shared_text_style_parameters,
    build_qfont_from_config,
    font_color_from_config,
    draw_styled_text,
)


class SimpleKaraokeVisualization(BaseLyricsVisualization):
    """
    Simple karaoke-style lyrics visualization.

    It renders:
      - the current phrase as a single line,
      - the active word highlighted in-place with a glow rectangle.

    The global background (solid / gradient / project cover) is rendered
    by this plugin using the shared background_* keys, in the same spirit
    as VerticalScrollLyricsVisualization.
    """

    plugin_id: str = "lyrics_karaoke_single_line"
    plugin_name: str = "Karaoke – single line highlight"
    plugin_description: str = (
        "Displays the current phrase as a single line and highlights "
        "the active word with an audio-reactive glow. Background can be "
        "the project cover, a gradient, or a solid color."
    )
    plugin_author: str = "Olaf"
    plugin_version: str = "1.1.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Ensure shared text-style defaults (font, outline, shadow, box, background)
        apply_default_text_style_config(self.config)

        # Smoothed amplitude for nicer visuals (avoid jitter)
        self._smoothed_amp: float = 0.0

        # Plugin-specific defaults
        self.config.setdefault("glow_strength", 0.7)
        self.config.setdefault("show_full_line", True)

        # Highlight geometry + color
        self.config.setdefault("highlight_padding_x", 0.4)
        self.config.setdefault("highlight_padding_y", 0.3)
        self.config.setdefault("highlight_color", "#ffff00")

        # Center position of the phrase in normalized coordinates (0–1)
        #  - center_x: 0 = fully left, 1 = fully right
        #  - center_y: 0 = top, 1 = bottom
        self.config.setdefault("center_x", 0.5)
        self.config.setdefault("center_y", 0.6)

        # Prefer the project cover as a background by default.
        self.config.setdefault("background_mode", "cover")

        # Reasonable minimum size for the preview widget
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Plugin parameters exposed to the host
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Declare all configurable parameters for this plugin.

        Font / outline / shadow / box / background settings are delegated
        to the shared text-style helper so that all installed fonts and
        the background options (cover / gradient / solid) are consistent
        across plugins.
        """
        # Shared text-style parameters (font family, size, bold, color,
        # outline, shadow, background box, background_mode, etc.)
        params = dict(shared_text_style_parameters())

        # Plugin-specific parameters (kept on top of shared ones)
        params.update(
            {
                # Phrase center position
                "center_x": PluginParameter(
                    name="center_x",
                    label="Phrase center – horizontal",
                    type="float",
                    default=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    description=(
                        "Horizontal position of the phrase center inside the frame "
                        "(0.0 = far left, 1.0 = far right)."
                    ),
                ),
                "center_y": PluginParameter(
                    name="center_y",
                    label="Phrase center – vertical",
                    type="float",
                    default=0.6,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    description=(
                        "Vertical position of the phrase center inside the frame "
                        "(0.0 = top, 1.0 = bottom)."
                    ),
                ),
                # Highlight controls
                "glow_strength": PluginParameter(
                    name="glow_strength",
                    label="Highlight glow strength",
                    type="float",
                    default=0.7,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Intensity of the glow around the active word.",
                ),
                "show_full_line": PluginParameter(
                    name="show_full_line",
                    label="Show full phrase",
                    type="bool",
                    default=True,
                    description="If unchecked, only the highlighted word is drawn.",
                ),
                "highlight_padding_x": PluginParameter(
                    name="highlight_padding_x",
                    label="Highlight padding (horizontal)",
                    type="float",
                    default=0.4,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description=(
                        "Horizontal padding around the active word highlight box, "
                        "in units of text height."
                    ),
                ),
                "highlight_padding_y": PluginParameter(
                    name="highlight_padding_y",
                    label="Highlight padding (vertical)",
                    type="float",
                    default=0.3,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description=(
                        "Vertical padding around the active word highlight box, "
                        "in units of text height."
                    ),
                ),
                "highlight_color": PluginParameter(
                    name="highlight_color",
                    label="Highlight color",
                    type="color",
                    default="#ffff00",
                    description="Fill color used behind the active word.",
                ),
            }
        )

        return params

    # ------------------------------------------------------------------
    # Frame update (called from the host with current time / amp / lyrics)
    # ------------------------------------------------------------------
    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Update internal state according to the current frame context.

        Here we mainly smooth the amplitude; the textual information
        (full line + active word) is read on demand in paintEvent via
        self.current_context().
        """
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.2  # exponential smoothing
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

    # ------------------------------------------------------------------
    # Background painting, aligned with VerticalScrollLyricsVisualization
    # ------------------------------------------------------------------
    def _paint_background(self, painter: QPainter, rect: QRectF) -> None:
        """
        Fill the global background using the shared background_* keys.

        Modes:
          - "cover": draw the project cover if available; otherwise fallback
                     to the gradient mode.
          - "solid": fill with config["background_color"].
          - "gradient" (default): vertical gradient that slightly reacts
                     to the audio amplitude.
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
                # Scale to cover the whole area while preserving aspect ratio
                scale = max(rw / pw, rh / ph)
                sw = int(pw * scale)
                sh = int(ph * scale)
                sx = int(rect.center().x() - sw / 2.0)
                sy = int(rect.center().y() - sh / 2.0)
                painter.drawPixmap(sx, sy, sw, sh, pix)
                return

        if mode == "solid":
            col_str = str(self.config.get("background_color", "#000000") or "#000000")
            color = QColor(col_str) if QColor.isValidColor(col_str) else QColor(0, 0, 0)
            painter.fillRect(rect, color)
            return

        # Default: gradient
        top_str = str(self.config.get("background_gradient_top", "#101010") or "#101010")
        bottom_str = str(
            self.config.get("background_gradient_bottom", "#402840") or "#402840"
        )
        top_color = QColor(top_str) if QColor.isValidColor(top_str) else QColor(16, 16, 16)
        bottom_color = (
            QColor(bottom_str) if QColor.isValidColor(bottom_str) else QColor(64, 40, 64)
        )

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
        """Render current phrase + active word highlight."""
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Global background (cover / gradient / solid)
        self._paint_background(painter, rect)

        ctx = self.current_context()
        if ctx is None or not ctx.text_full_line:
            # No lyrics: render nothing (background only).
            painter.end()
            return

        # Raw text from context
        line_text = ctx.text_full_line
        active_word = ctx.text_active_word or ""

        # Optional full capitalization
        capitalize = bool(self.config.get("capitalize_all", False))
        display_line = line_text.upper() if capitalize else line_text
        display_active_word = active_word.upper() if capitalize else active_word

        # ----- Font selection (family, size, bold) ----------------------
        try:
            font_size = int(self.config.get("font_size", 40))
        except Exception:
            font_size = 40
        font_size = max(8, font_size)

        base_font = build_qfont_from_config(self.config, painter, font_size)
        painter.setFont(base_font)
        fm = QFontMetrics(base_font)

        # Dynamic shrink to keep the full line inside the widget width
        max_width = rect.width() * 0.9
        while fm.horizontalAdvance(display_line) > max_width and font_size > 8:
            font_size = max(8, int(font_size * 0.9))
            base_font = build_qfont_from_config(self.config, painter, font_size)
            painter.setFont(base_font)
            fm = QFontMetrics(base_font)

        font_color = font_color_from_config(self.config)

        # ------------------------------------------------------------------
        # Compute phrase center from sliders (normalized 0–1)
        # ------------------------------------------------------------------
        def _clamp01(value: float, default: float) -> float:
            try:
                v = float(value)
            except Exception:
                v = default
            return max(0.0, min(1.0, v))

        center_x_frac = _clamp01(self.config.get("center_x", 0.5), 0.5)
        center_y_frac = _clamp01(self.config.get("center_y", 0.6), 0.6)

        line_width = fm.horizontalAdvance(display_line)

        # Horizontal center in pixels
        center_x_px = rect.left() + center_x_frac * rect.width()
        # Baseline Y from top
        baseline_y = rect.top() + center_y_frac * rect.height()

        # X of the beginning of the line
        line_x0 = center_x_px - line_width / 2.0

        # ------------------------------------------------------------------
        # Draw the full phrase (with background box / outline / shadow)
        # ------------------------------------------------------------------
        if bool(self.config.get("show_full_line", True)):
            # Slightly transparent color for the base line, so that the
            # active word can stand out on top.
            line_color = QColor(font_color)
            line_color.setAlpha(190)

            draw_styled_text(
                painter=painter,
                x=line_x0,
                y=baseline_y,
                text=display_line,
                config=self.config,
                base_font=base_font,
                base_color=line_color,
            )

        # ------------------------------------------------------------------
        # Active word highlight
        # ------------------------------------------------------------------
        idx_start = ctx.word_char_start
        idx_end = ctx.word_char_end

        if display_active_word.strip() and (idx_start is not None) and (idx_start >= 0):
            # Use alignment-based character span if available
            if idx_end is None or idx_end <= idx_start:
                idx_end = idx_start + len(display_active_word)

            idx_start = max(0, min(idx_start, len(display_line)))
            idx_end = max(idx_start, min(idx_end, len(display_line)))

            prefix = display_line[:idx_start]
            word_text = display_line[idx_start:idx_end]
        else:
            # Fallback: simple first occurrence search
            if display_active_word.strip():
                lower_line = display_line.lower()
                lower_word = display_active_word.lower()
                idx = lower_line.find(lower_word)
            else:
                idx = -1

            if idx < 0:
                # Nothing to highlight
                painter.end()
                return

            prefix = display_line[:idx]
            word_text = display_line[idx : idx + len(display_active_word)]

        prefix_width = fm.horizontalAdvance(prefix)
        word_width = fm.horizontalAdvance(word_text)
        word_x = line_x0 + prefix_width

        # Compute highlight rectangle around the active word
        glow_strength = float(self.config.get("glow_strength", 0.7))
        glow_strength = max(0.0, min(1.0, glow_strength))
        glow = self._smoothed_amp * glow_strength

        # Highlight geometry factors
        try:
            pad_x_factor = float(self.config.get("highlight_padding_x", 0.4))
        except Exception:
            pad_x_factor = 0.4
        try:
            pad_y_factor = float(self.config.get("highlight_padding_y", 0.3))
        except Exception:
            pad_y_factor = 0.3
        pad_x_factor = max(0.0, pad_x_factor)
        pad_y_factor = max(0.0, pad_y_factor)

        padding_x = fm.height() * pad_x_factor
        padding_y = fm.height() * pad_y_factor

        # Highlight color from config (hex), with audio-reactive alpha
        col_str = str(self.config.get("highlight_color", "#ffff00") or "#ffff00")
        bg_color = QColor(col_str) if QColor.isValidColor(col_str) else QColor(255, 255, 0)
        bg_color.setAlphaF(0.25 + 0.55 * glow)

        highlight_rect = QRectF(
            word_x - padding_x,
            baseline_y - fm.ascent() - padding_y,
            word_width + 2.0 * padding_x,
            fm.height() + 2.0 * padding_y,
        )

        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(highlight_rect, 8.0, 8.0)

        # ------------------------------------------------------------------
        # Draw the active word on top, without re-drawing a box
        # ------------------------------------------------------------------
        word_config: Dict[str, Any] = dict(self.config)
        # Ensure we do NOT draw a second box for the word only
        word_config["text_box_enabled"] = False

        draw_styled_text(
            painter=painter,
            x=word_x,
            y=baseline_y,
            text=word_text,
            config=word_config,
            base_font=base_font,
            base_color=font_color,
        )

        painter.end()
