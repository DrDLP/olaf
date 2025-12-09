from __future__ import annotations

from typing import Any, Dict, Optional
import random

from PyQt6.QtGui import QPainter, QColor, QFontMetrics
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


class BigWordPulseVisualization(BaseLyricsVisualization):
    """
    Display a single word at the center of the screen, reacting to the
    vocal amplitude.

    Behaviour:
      - Only the currently active word is shown.
      - Its size scales with the audio amplitude.
      - Each new word gets a small random tilt (rotation) for a "collage" effect.

    Styling (font, outline, shadow, background box, capitalization)
    is provided by the shared text-style helpers.
    """

    plugin_id: str = "lyrics_big_word_pulse"
    plugin_name: str = "Big word pulse"
    plugin_description: str = (
        "Shows a single word at the center, scaling with the vocal amplitude "
        "and slightly rotating on each word change."
    )
    plugin_author: str = "Olaf"
    plugin_version: str = "0.4.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Shared text-style defaults
        apply_default_text_style_config(self.config)

        # Audio / motion state
        self._smoothed_amp: float = 0.0
        self._current_word: str = ""
        self._current_angle_deg: float = 0.0
        self._rng = random.Random()

        # Plugin-specific defaults
        self.config.setdefault("base_font_size", 64)
        self.config.setdefault("amp_scale", 1.0)
        self.config.setdefault("min_scale", 0.8)
        self.config.setdefault("max_scale", 2.0)
        self.config.setdefault("center_y", 0.55)
        self.config.setdefault("rotation_variation_deg", 6.0)

        # Background configuration (same convention as other lyrics plugins)
        self.config.setdefault("background_mode", "gradient")  # "gradient", "solid", "cover"
        self.config.setdefault("background_color", "#000000")
        self.config.setdefault("background_gradient_top", "#101010")
        self.config.setdefault("background_gradient_bottom", "#402840")

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
        params = text_style_parameters()
        params.update(
            {
                "background_mode": PluginParameter(
                    name="background_mode",
                    label="Background mode",
                    type="enum",
                    default="gradient",
                    choices=["gradient", "solid", "cover"],
                    description="How to fill the background: gradient, solid color, or project cover.",
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
                "base_font_size": PluginParameter(
                    name="base_font_size",
                    label="Base font size",
                    type="int",
                    default=64,
                    minimum=16,
                    maximum=200,
                    step=1,
                    description="Base point size before applying audio scaling.",
                ),
                "amp_scale": PluginParameter(
                    name="amp_scale",
                    label="Amplitude scale",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    description=(
                        "How strongly the audio amplitude changes the word size. "
                        "0 = no scaling."
                    ),
                ),
                "min_scale": PluginParameter(
                    name="min_scale",
                    label="Minimum scale",
                    type="float",
                    default=0.8,
                    minimum=0.1,
                    maximum=3.0,
                    step=0.1,
                    description="Lower bound on the size multiplier.",
                ),
                "max_scale": PluginParameter(
                    name="max_scale",
                    label="Maximum scale",
                    type="float",
                    default=2.0,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1,
                    description="Upper bound on the size multiplier.",
                ),
                "center_y": PluginParameter(
                    name="center_y",
                    label="Vertical center (rel.)",
                    type="float",
                    default=0.55,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    description="Relative vertical position of the word center.",
                ),
                "rotation_variation_deg": PluginParameter(
                    name="rotation_variation_deg",
                    label="Rotation variation (°)",
                    type="float",
                    default=6.0,
                    minimum=0.0,
                    maximum=45.0,
                    step=0.5,
                    description=(
                        "Maximum absolute tilt applied when the word changes "
                        "(random in [-var, +var])."
                    ),
                ),
            }
        )
        return params

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------
    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Smooth amplitude and detect word changes to update the tilt.
        """
        # Smooth amplitude
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.25
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

        # Detect word change
        new_word = (ctx.text_active_word or "").strip()
        if new_word != self._current_word:
            self._current_word = new_word
            # New random small rotation
            try:
                max_deg = float(self.config.get("rotation_variation_deg", 6.0))
            except Exception:
                max_deg = 6.0
            max_deg = max(0.0, max_deg)
            if max_deg > 0.0:
                self._current_angle_deg = self._rng.uniform(-max_deg, max_deg)
            else:
                self._current_angle_deg = 0.0

    # ------------------------------------------------------------------
    # Background helper (same convention as karaoke plugin)
    # ------------------------------------------------------------------
    def _paint_background(self, painter: QPainter, rect) -> None:
        mode = str(self.config.get("background_mode", "gradient") or "gradient")

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
                return

        if mode == "solid":
            col_str = str(self.config.get("background_color", "#000000") or "#000000")
            color = QColor(col_str) if QColor.isValidColor(col_str) else QColor(0, 0, 0)
            painter.fillRect(rect, color)
            return

        # Default gradient
        top_str = str(self.config.get("background_gradient_top", "#101010") or "#101010")
        bottom_str = str(self.config.get("background_gradient_bottom", "#402840") or "#402840")
        top_color = QColor(top_str) if QColor.isValidColor(top_str) else QColor(16, 16, 16)
        bottom_color = QColor(bottom_str) if QColor.isValidColor(bottom_str) else QColor(64, 40, 64)

        bottom_color = QColor(
            min(255, bottom_color.red() + int(80 * amp)),
            bottom_color.green(),
            min(255, bottom_color.blue() + int(80 * amp)),
        )

        from PyQt6.QtGui import QLinearGradient, QBrush  # local import to avoid clutter at top

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

        rect = self.rect()

        # Background: according to shared config (solid / gradient / cover)
        self._paint_background(painter, rect)

        ctx = self.current_context()
        if ctx is None or not (ctx.text_active_word or "").strip():
            # Fallback message
            painter.setPen(QColor(200, 200, 200))
            font = painter.font()
            font.setPointSize(18)
            painter.setFont(font)
            msg = "Waiting for word."
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(msg)
            x = rect.center().x() - tw / 2.0
            y = rect.center().y()
            painter.drawText(int(x), int(y), msg)
            painter.end()
            return

        raw_word = (ctx.text_active_word or "").strip()
        capitalize = bool(self.config.get("capitalize_all", False))
        word_text = raw_word.upper() if capitalize else raw_word

        # ------------------------------------------------------------------
        # Compute scaled font size from amplitude
        # ------------------------------------------------------------------
        try:
            base_size = int(self.config.get("base_font_size", 64))
        except Exception:
            base_size = 64
        base_size = max(8, base_size)

        try:
            amp_scale = float(self.config.get("amp_scale", 1.0))
        except Exception:
            amp_scale = 1.0

        try:
            min_scale = float(self.config.get("min_scale", 0.8))
        except Exception:
            min_scale = 0.8
        try:
            max_scale = float(self.config.get("max_scale", 2.0))
        except Exception:
            max_scale = 2.0

        min_scale = max(0.1, min_scale)
        max_scale = max(min_scale, max_scale)

        # scale factor in [min_scale, max_scale], proportional to smoothed amp
        raw_scale = min_scale + self._smoothed_amp * amp_scale
        scale = max(min_scale, min(raw_scale, max_scale))

        font_size = int(base_size * scale)
        font_size = max(8, font_size)

        base_font = build_qfont_from_config(self.config, painter, font_size)
        painter.setFont(base_font)
        fm = QFontMetrics(base_font)

        # Dynamic shrink to avoid word exiting the screen horizontally
        max_width = rect.width() * 0.9
        while fm.horizontalAdvance(word_text) > max_width and font_size > 8:
            font_size = max(8, int(font_size * 0.9))
            base_font = build_qfont_from_config(self.config, painter, font_size)
            painter.setFont(base_font)
            fm = QFontMetrics(base_font)

        font_color = font_color_from_config(self.config)

        text_width = fm.horizontalAdvance(word_text)

        cx = rect.center().x()

        try:
            center_y_rel = float(self.config.get("center_y", 0.55))
        except Exception:
            center_y_rel = 0.55
        center_y_rel = max(0.0, min(1.0, center_y_rel))

        cy = rect.height() * center_y_rel

        # Baseline such that the text is vertically centered around cy
        ascent = fm.ascent()
        descent = fm.descent()
        baseline_y = cy + (ascent - descent) / 2.0

        word_x = cx - text_width / 2.0

        # ------------------------------------------------------------------
        # Apply rotation around the center and draw styled text
        # ------------------------------------------------------------------
        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self._current_angle_deg)
        painter.translate(-cx, -cy)

        draw_styled_text(
            painter=painter,
            x=word_x,
            y=baseline_y,
            text=word_text,
            config=self.config,
            base_font=base_font,
            base_color=font_color,
        )

        painter.restore()
        painter.end()
