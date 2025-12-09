from __future__ import annotations

"""
Text Waveform Lyrics Visualization.

Each word of the current line is placed along a horizontal line and
animated vertically on a sinusoidal path driven by the audio amplitude.

- Words are laid out from left to right as normal text.
- The vertical offset of each word follows: y = y_base + sin(phase) * A
  where phase depends on the word index and global time t.
- The active word is slightly more pronounced (greater amplitude).
"""

from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QFontMetrics, QLinearGradient, QBrush
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


class TextWaveformVisualization(BaseLyricsVisualization):
    """
    Words laid out horizontally, bouncing on a sinusoidal "waveform".
    """

    plugin_id: str = "text_waveform"
    plugin_name: str = "Text waveform"
    plugin_description: str = (
        "Displays each word as a point on a horizontal text waveform, "
        "animated vertically by the audio amplitude."
    )
    plugin_author: str = "App Olaf"
    plugin_version: str = "0.1.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Make sure shared text-style keys exist
        apply_default_text_style_config(self.config)

        # Use project cover as default background, same behaviour
        # as in other lyrics plugins.
        self.config.setdefault("background_mode", "cover")

        # Audio state
        self._smoothed_amp: float = 0.0

        # Waveform behaviour
        self.config.setdefault("wave_amplitude", 0.6)          # in multiples of font size
        self.config.setdefault("wave_frequency", 1.1)          # phase increment per word
        self.config.setdefault("wave_speed", 1.5)              # speed factor for time t
        self.config.setdefault("active_amplitude_boost", 0.4)  # extra amplitude for active word

        # Horizontal layout
        self.config.setdefault("horizontal_margin", 0.08)      # fraction of widget width

        # Visual emphasis of past / future / active words
        self.config.setdefault("past_words_alpha", 1.0)
        self.config.setdefault("future_words_alpha", 0.55)
        self.config.setdefault("active_words_alpha", 1.0)

        # Reasonable widget size
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose shared text-style parameters + plugin-specific controls.
        """
        params = dict(text_style_parameters())

        params.update(
            {
                # Background (same semantics as other plugins)
                "background_mode": PluginParameter(
                    name="background_mode",
                    label="Background mode",
                    type="enum",
                    default="cover",
                    choices=["cover", "solid", "gradient"],
                    description="How to fill the background: cover, solid color, or gradient.",
                ),
                "background_color": PluginParameter(
                    name="background_color",
                    label="Background color",
                    type="color",
                    default="#000000",
                    description="Background color when mode is 'solid'.",
                ),
                "background_gradient_top": PluginParameter(
                    name="background_gradient_top",
                    label="Gradient top color",
                    type="color",
                    default="#101010",
                    description="Top color of the vertical gradient background.",
                ),
                "background_gradient_bottom": PluginParameter(
                    name="background_gradient_bottom",
                    label="Gradient bottom color",
                    type="color",
                    default="#402840",
                    description="Bottom color of the vertical gradient background.",
                ),
                # Wave parameters
                "wave_amplitude": PluginParameter(
                    name="wave_amplitude",
                    label="Wave amplitude (× font size)",
                    type="float",
                    default=0.6,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Base vertical amplitude of the waveform, in multiples of the font size.",
                ),
                "wave_frequency": PluginParameter(
                    name="wave_frequency",
                    label="Wave frequency (per word)",
                    type="float",
                    default=1.1,
                    minimum=0.1,
                    maximum=4.0,
                    step=0.1,
                    description="Phase increment per word in the sine function.",
                ),
                "wave_speed": PluginParameter(
                    name="wave_speed",
                    label="Wave speed",
                    type="float",
                    default=1.5,
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    description="Horizontal speed of the wave over time.",
                ),
                "active_amplitude_boost": PluginParameter(
                    name="active_amplitude_boost",
                    label="Active word amplitude boost",
                    type="float",
                    default=0.4,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Extra amplitude factor applied to the active word.",
                ),
                "horizontal_margin": PluginParameter(
                    name="horizontal_margin",
                    label="Horizontal margin (fraction)",
                    type="float",
                    default=0.08,
                    minimum=0.0,
                    maximum=0.4,
                    step=0.01,
                    description="Left/right margin as a fraction of the widget width.",
                ),
                # Alpha for word groups
                "past_words_alpha": PluginParameter(
                    name="past_words_alpha",
                    label="Past words alpha",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for words already sung.",
                ),
                "future_words_alpha": PluginParameter(
                    name="future_words_alpha",
                    label="Future words alpha",
                    type="float",
                    default=0.55,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for upcoming words.",
                ),
                "active_words_alpha": PluginParameter(
                    name="active_words_alpha",
                    label="Active word alpha",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for the active word.",
                ),
            }
        )

        return params

    # ------------------------------------------------------------------
    # Background helper (same convention as vertical_scroll / double_line)
    # ------------------------------------------------------------------
    def _paint_background(self, painter: QPainter, rect) -> None:
        """
        Fill the background with cover / solid / gradient.
        """
        mode = str(self.config.get("background_mode", "cover") or "cover")

        # Try to retrieve a cover pixmap if the host provides one.
        cover_source = getattr(self, "cover_pixmap", None)
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
    # Frame update
    # ------------------------------------------------------------------
    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Smooth amplitude. The rest of the state is read directly
        from the context in paintEvent().
        """
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.2
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

    # ------------------------------------------------------------------
    # Word splitting helper (like in vertical_scroll, but local)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_words_with_indices(text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into (word, start_char, end_char) using whitespace
        as separator. The indices refer to the original string.
        """
        results: List[Tuple[str, int, int]] = []
        if not text:
            return results

        start = 0
        length = len(text)
        i = 0
        while i < length:
            # Skip leading spaces
            while i < length and text[i].isspace():
                i += 1
            if i >= length:
                break
            word_start = i
            while i < length and not text[i].isspace():
                i += 1
            word_end = i
            word = text[word_start:word_end]
            if word:
                results.append((word, word_start, word_end))
        return results

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:  # type: ignore[override]
        """
        Render:
          - background (cover / solid / gradient),
          - each word of the current line placed along a horizontal line,
          - each word following a sinusoidal vertical offset based on
            audio amplitude and time.
        """
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Background first
        self._paint_background(painter, rect)

        ctx = self.current_context()
        if ctx is None:
            painter.end()
            return

        raw_line = (ctx.text_full_line or "").strip()
        if not raw_line:
            # No lyrics for this frame: background only
            painter.end()
            return

        # --------------------------------------------------------------
        # Base typography
        # --------------------------------------------------------------
        try:
            base_point_size = int(self.config.get("font_size", 40))
        except Exception:
            base_point_size = 40
        base_point_size = max(8, base_point_size)

        base_font = build_qfont_from_config(self.config, painter, base_point_size)
        painter.setFont(base_font)
        base_color = font_color_from_config(self.config)

        fm = QFontMetrics(base_font)

        # Capitalization for visual text only
        capitalize_all = bool(self.config.get("capitalize_all", False))

        # Words with char indices (on raw string)
        words_info = self._split_words_with_indices(raw_line)
        if not words_info:
            painter.end()
            return

        # Active word range from context
        word_char_start = getattr(ctx, "word_char_start", None)
        word_char_end = getattr(ctx, "word_char_end", None)

        # --------------------------------------------------------------
        # Horizontal layout: compute total width and center
        # --------------------------------------------------------------
        # We'll layout words one after another with spaces, and then
        # shift the whole line so it is centered.
        words_visual: List[str] = []
        for word, _, _ in words_info:
            w = word.upper() if capitalize_all else word
            words_visual.append(w)

        # Approximate total width including single spaces between words
        space_width = fm.horizontalAdvance(" ")
        total_width = 0
        for idx, w in enumerate(words_visual):
            total_width += fm.horizontalAdvance(w)
            if idx < len(words_visual) - 1:
                total_width += space_width

        # Apply horizontal margins
        margin_frac = float(self.config.get("horizontal_margin", 0.08))
        margin_frac = max(0.0, min(0.4, margin_frac))

        usable_width = rect.width() * (1.0 - 2.0 * margin_frac)
        x_offset = rect.left() + rect.width() * margin_frac

        # If the line is wider than the usable area, we still center the
        # text, it may go closer to the edges.
        if total_width < usable_width:
            line_left = x_offset + (usable_width - total_width) / 2.0
        else:
            line_left = rect.center().x() - total_width / 2.0

        center_y = rect.center().y()
        baseline_y = center_y

        # Wave parameters
        wave_amp_factor = float(self.config.get("wave_amplitude", 0.6))
        wave_freq = float(self.config.get("wave_frequency", 1.1))
        wave_speed = float(self.config.get("wave_speed", 1.5))
        active_boost = float(self.config.get("active_amplitude_boost", 0.4))

        amp = self._smoothed_amp
        # Convert amplitude in units of font size to pixels
        base_amp_px = wave_amp_factor * float(base_point_size)

        # Time from context for phase evolution
        t = getattr(ctx, "t", 0.0) or 0.0

        # Alpha configuration
        past_alpha = float(self.config.get("past_words_alpha", 1.0))
        future_alpha = float(self.config.get("future_words_alpha", 0.55))
        active_alpha = float(self.config.get("active_words_alpha", 1.0))

        # --------------------------------------------------------------
        # Draw each word on the waveform
        # --------------------------------------------------------------
        cursor_x = float(line_left)

        for idx, (raw_word, start_i, end_i) in enumerate(words_info):
            visual_word = words_visual[idx]
            word_width = fm.horizontalAdvance(visual_word)

            # Determine word "state" relative to the active char range
            is_active = (
                isinstance(word_char_start, int)
                and isinstance(word_char_end, int)
                and word_char_start is not None
                and word_char_end is not None
                and start_i <= word_char_start < end_i
            )
            is_past = (
                isinstance(word_char_start, int)
                and word_char_start is not None
                and end_i <= word_char_start
            )
            is_future = (
                isinstance(word_char_end, int)
                and word_char_end is not None
                and start_i >= word_char_end
            )

            # Phase for this word
            phase = idx * wave_freq + t * wave_speed

            # Extra amplitude for the active word
            local_amp_px = base_amp_px * amp
            if is_active:
                local_amp_px *= (1.0 + active_boost)

            # Vertical offset: baseline + sin(phase) * amplitude
            y_offset = local_amp_px * (0.0 if local_amp_px == 0 else __import__("math").sin(phase))
            word_baseline_y = baseline_y + y_offset

            # Set opacity by state
            if is_active:
                alpha = active_alpha
            elif is_past:
                alpha = past_alpha
            elif is_future:
                alpha = future_alpha
            else:
                # If we cannot classify, treat as current/past
                alpha = past_alpha

            alpha = max(0.0, min(1.0, alpha))

            painter.save()
            painter.setFont(base_font)
            painter.setOpacity(alpha)

            # We draw each word individually. For the waveform effect,
            # a simple style (no box / shadow) is often cleaner; we
            # keep the global style but you can override it here if needed.
            local_config = dict(self.config)

            # Optional: if you want absolutely no box per word, uncomment:
            # local_config["text_box_enabled"] = False

            draw_styled_text(
                painter=painter,
                x=cursor_x,
                y=float(word_baseline_y),
                text=visual_word,
                config=local_config,
                base_font=base_font,
                base_color=base_color,
            )

            painter.restore()

            # Advance cursor (word + space)
            cursor_x += word_width + space_width

        painter.end()
