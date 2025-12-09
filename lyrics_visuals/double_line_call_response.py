from __future__ import annotations

"""
Double-line "call & response" lyrics visualization.

This plugin displays:
  - the *previous* phrase as a small, semi-transparent line above,
  - the *current* phrase centered in the middle,
  - the active word of the current phrase highlighted with a soft glow.

It is intended for dense vocals (rap, fast pop, etc.) where keeping
the previous line in view helps comprehension.
"""

from typing import Any, Dict, Optional

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


class DoubleLineCallResponseVisualization(BaseLyricsVisualization):
    """
    Previous line on top, current line in the center, active word highlighted.
    """

    plugin_id: str = "double_line_call_response"
    plugin_name: str = "Double-line (call & response)"
    plugin_description: str = (
        "Shows the previous phrase above the current one, "
        "with the active word highlighted on the current line."
    )
    plugin_author: str = "DrDLP"
    plugin_version: str = "1.1.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Ensure all shared text-style keys exist in the config.
        apply_default_text_style_config(self.config)

        # Default background: use project cover (same behaviour as vertical_scroll)
        self.config.setdefault("background_mode", "cover")

        # Audio / state
        self._smoothed_amp: float = 0.0
        self._current_phrase_index: Optional[int] = None
        self._current_line_text: str = ""
        self._previous_line_text: str = ""

        # Track whether we have ever had a real "previous phrase".
        # This is used to avoid drawing a faded line above the very
        # first phrase (and in other edge cases).
        self._has_real_previous: bool = False

        # Slide animation state (phrase transition)
        self._is_sliding: bool = False
        self._slide_start_t: Optional[float] = None
        self._sliding_outgoing_line: str = ""  # line leaving the center
        self._sliding_incoming_line: str = ""  # new line entering the center

        # Frozen geometry for the current phrase slide
        # (so that later layout changes do not affect an ongoing slide).
        self._slide_center_y: Optional[float] = None
        self._slide_gap: Optional[float] = None

        # Silence fade-out state (when there is a gap with no next phrase yet)
        self._silence_active: bool = False
        self._silence_start_t: Optional[float] = None
        self._silence_last_line: str = ""

        # Plugin-specific defaults
        self.config.setdefault("previous_line_scale", 0.7)
        self.config.setdefault("previous_line_alpha", 1)
        self.config.setdefault("vertical_spacing_factor", 1.1)

        self.config.setdefault("highlight_color", "#ffff80")
        self.config.setdefault("highlight_padding_x", 8.0)
        self.config.setdefault("highlight_padding_y", 4.0)
        self.config.setdefault("highlight_glow_strength", 0.9)

        # Duration of the slide when a phrase changes (seconds)
        self.config.setdefault("phrase_slide_duration", 0.1)

        # Duration of the fade/slide when there is a silence (seconds)
        self.config.setdefault("silence_fade_duration", 0.6)

        # Reasonable widget size and resizing behaviour
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
                "previous_line_scale": PluginParameter(
                    name="previous_line_scale",
                    label="Previous line scale",
                    type="float",
                    default=0.7,
                    minimum=0.2,
                    maximum=1.5,
                    step=0.05,
                    description="Scale factor applied to the previous line font size.",
                ),
                "previous_line_alpha": PluginParameter(
                    name="previous_line_alpha",
                    label="Previous line alpha",
                    type="float",
                    default=0.1,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity of the previous line (0 = invisible, 1 = fully opaque).",
                ),
                "vertical_spacing_factor": PluginParameter(
                    name="vertical_spacing_factor",
                    label="Vertical spacing",
                    type="float",
                    default=1.1,
                    minimum=0.5,
                    maximum=2.0,
                    step=0.05,
                    description="Vertical spacing between previous and current line (in line heights).",
                ),
                "phrase_slide_duration": PluginParameter(
                    name="phrase_slide_duration",
                    label="Phrase slide duration (s)",
                    type="float",
                    default=0.1,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Duration of the vertical slide when the phrase changes (0 = no slide).",
                ),
                "silence_fade_duration": PluginParameter(
                    name="silence_fade_duration",
                    label="Silence fade duration (s)",
                    type="float",
                    default=0.6,
                    minimum=0.1,
                    maximum=3.0,
                    step=0.05,
                    description="Fade/slide duration for the last line during silent gaps.",
                ),
                # Highlight of the active word
                "highlight_color": PluginParameter(
                    name="highlight_color",
                    label="Highlight color",
                    type="color",
                    default="#ffff80",
                    description="Fill color used behind the active word.",
                ),
                "highlight_padding_x": PluginParameter(
                    name="highlight_padding_x",
                    label="Highlight padding X",
                    type="float",
                    default=8.0,
                    minimum=0.0,
                    maximum=80.0,
                    step=1.0,
                    description="Horizontal padding around the active word highlight box.",
                ),
                "highlight_padding_y": PluginParameter(
                    name="highlight_padding_y",
                    label="Highlight padding Y",
                    type="float",
                    default=4.0,
                    minimum=0.0,
                    maximum=80.0,
                    step=1.0,
                    description="Vertical padding around the active word highlight box.",
                ),
                "highlight_glow_strength": PluginParameter(
                    name="highlight_glow_strength",
                    label="Highlight glow strength",
                    type="float",
                    default=0.9,
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    description="How strongly the audio amplitude modulates the highlight glow.",
                ),
            }
        )

        return params

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------
    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Smooth amplitude and track phrase transitions.

        - Keeps a stable "previous line" above the current one.
        - Starts a slide animation when phrase_index changes.
        - Detects silence gaps (no text) and prepares a fade/slide of the
          last sung phrase even if there is no next phrase yet.
        """
        # --- Amplitude smoothing ------------------------------------------------
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.2
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

        phrase_idx = ctx.phrase_index
        line_now = (ctx.text_full_line or "").strip()

        # --- Silence handling: no text on this frame ---------------------------
        if not line_now:
            # If we just entered a silence and we had a visible line before,
            # remember it for a fade/slide-out and remove it from the
            # "previous line" state so it will not be reused later.
            if (not self._silence_active) and self._current_line_text.strip():
                self._silence_active = True
                self._silence_start_t = getattr(ctx, "t", 0.0)
                self._silence_last_line = self._current_line_text.strip()

                # IMPORTANT:
                # This line is now handled exclusively by the silence fade.
                # We clear previous/current caches so it will never reappear
                # as a contextual "previous line" after the fade has finished.
                self._previous_line_text = ""
                self._current_line_text = ""

            # During silence, do not update phrase index / previous line:
            # we only keep amplitude moving.
            return

        else:
            # We have lyrics again: any silence fade-out is over.
            self._silence_active = False
            self._silence_start_t = None
            self._silence_last_line = ""

        # --- Normal phrase tracking (with text present) ------------------------
        if phrase_idx is None:
            # No phrase info: keep current text, only amp was updated.
            return

        if self._current_phrase_index is None:
            # First phrase seen.
            self._current_phrase_index = phrase_idx
            self._current_line_text = line_now
            self._previous_line_text = ""
            self._is_sliding = False
            self._slide_start_t = None
            self._sliding_outgoing_line = ""
            self._sliding_incoming_line = ""
            return

        if phrase_idx != self._current_phrase_index:
            # Phrase index changed: prepare slide from old center line to new one.
            outgoing = (self._current_line_text or "").strip()
            incoming = line_now

            self._current_phrase_index = phrase_idx
            self._current_line_text = incoming

            if outgoing:
                # Outgoing line becomes the "previous" one after the slide.
                self._previous_line_text = outgoing
                # From now on we have a genuine previous phrase.
                self._has_real_previous = True

            # Configure slide animation only if both lines are non-empty.
            if outgoing and incoming:
                self._sliding_outgoing_line = outgoing
                self._sliding_incoming_line = incoming
                self._slide_start_t = getattr(ctx, "t", 0.0)
                self._is_sliding = True
                # Reset frozen geometry; it will be captured in paintEvent
                # on the first frame of this slide.
                self._slide_center_y = None
                self._slide_gap = None

            else:
                self._is_sliding = False
                self._slide_start_t = None
                self._sliding_outgoing_line = ""
                self._sliding_incoming_line = ""
        else:
            # Same phrase: keep the current line text in sync.
            self._current_line_text = line_now or self._current_line_text

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------
 
    def _paint_background(self, painter: QPainter, rect) -> None:
        """
        Fill the global background (outside / behind the lyrics lines).

        Same behaviour as VerticalScrollLyricsVisualization:
        - 'cover': project cover if available
        - 'solid': plain color
        - 'gradient': vertical gradient with slight amp reactivity
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


    def paintEvent(self, event) -> None:  # type: ignore[override]
        """
        Render:
          - background cover / gradient,
          - previous phrase (small, faded, above),
          - current phrase centered,
          - active word glow on the current phrase,
          - slide animation on phrase change,
          - fade/slide-out of the last phrase during silences.
        """
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Background first (cover / solid / gradient, shared with other plugins)
        self._paint_background(painter, rect)

        ctx = self.current_context()
        if ctx is None:
            # No timing / lyrics context: background only.
            painter.end()
            return

        # --------------------------------------------------------------
        # Base typography (shared text-style configuration)
        # --------------------------------------------------------------
        try:
            base_point_size = int(self.config.get("font_size", 40))
        except Exception:
            base_point_size = 40
        base_point_size = max(8, base_point_size)

        base_font = build_qfont_from_config(self.config, painter, base_point_size)
        painter.setFont(base_font)
        base_color = font_color_from_config(self.config)

        fm_current = QFontMetrics(base_font)

        # Previous line font is a scaled version of the base font.
        prev_scale = float(self.config.get("previous_line_scale", 0.7))
        prev_scale = max(0.1, prev_scale)
        prev_font = base_font
        if prev_scale != 1.0:
            prev_font = type(base_font)(base_font)
            prev_font.setPointSize(int(base_point_size * prev_scale))
        fm_prev = QFontMetrics(prev_font)

        # --------------------------------------------------------------
        # Resolve texts for current / previous lines and slide state
        # --------------------------------------------------------------
        raw_ctx_line = (ctx.text_full_line or "").strip()
        has_ctx_line = bool(raw_ctx_line)

        # Default: use current context line (or cached) + stored previous line.
        raw_current_line = raw_ctx_line or self._current_line_text.strip()
        raw_previous_line = self._previous_line_text.strip()

        raw_current_for_frame = raw_current_line
        raw_prev_for_frame = raw_previous_line

        # Silence state: no text in the context but we have a line to fade out.
        is_silence = (
            self._silence_active
            and not has_ctx_line
            and bool(self._silence_last_line.strip())
        )

        # Slide configuration (for normal phrase transitions only)
        try:
            slide_duration = float(self.config.get("phrase_slide_duration", 0.35))
        except Exception:
            slide_duration = 0.35
        slide_duration = max(0.0, slide_duration)

        sliding = False
        slide_progress = 1.0
        t = getattr(ctx, "t", None)

        if (
            self._is_sliding
            and not is_silence  # no mirror slide during silence fade-out
            and self._slide_start_t is not None
            and slide_duration > 0.0
            and t is not None
        ):
            dt = t - self._slide_start_t
            if dt <= 0.0:
                slide_progress = 0.0
                sliding = True
            elif dt < slide_duration:
                slide_progress = max(0.0, min(1.0, dt / slide_duration))
                sliding = True
            else:
                # Slide finished, switch to static layout
                self._is_sliding = False
                self._slide_start_t = None
                self._sliding_outgoing_line = ""
                self._sliding_incoming_line = ""
                self._slide_center_y = None
                self._slide_gap = None
                slide_progress = 1.0

            if sliding:
                # During the slide, show outgoing line in the "previous" slot
                # and incoming line in the "current" slot (mirror movement).
                if self._sliding_incoming_line:
                    raw_current_for_frame = self._sliding_incoming_line
                if self._sliding_outgoing_line:
                    raw_prev_for_frame = self._sliding_outgoing_line

        # Silence override: during a gap, only the last sung line is shown
        # as "previous", sliding/fading upwards, with no current line.
        if is_silence:
            raw_prev_for_frame = self._silence_last_line.strip()
            raw_current_for_frame = ""

        # Capitalization
        if bool(self.config.get("capitalize_all", False)):
            visual_current_line = raw_current_for_frame.upper()
            visual_previous_line = raw_prev_for_frame.upper()
        else:
            visual_current_line = raw_current_for_frame
            visual_previous_line = raw_prev_for_frame

        # --------------------------------------------------------------
        # Phrase timing / fade factors
        # --------------------------------------------------------------
        # Normal case: previous line fades over the *current* phrase duration.
        try:
            phrase_duration = float(getattr(ctx, "phrase_duration", 0.0) or 0.0)
        except Exception:
            phrase_duration = 0.0
        phrase_duration = max(0.0, phrase_duration)

        try:
            local_t = float(getattr(ctx, "local_phrase_time", 0.0) or 0.0)
        except Exception:
            local_t = 0.0
        local_t = max(0.0, local_t)

        fade_prev = 1.0
        if phrase_duration > 0.0 and not is_silence:
            fade_prev = 1.0 - (local_t / phrase_duration)
            fade_prev = max(0.0, min(1.0, fade_prev))

        # Silence fade-out: use time since _silence_start_t instead.
        if is_silence and self._silence_start_t is not None and t is not None:
            dt_silence = max(0.0, t - self._silence_start_t)
            # Use dedicated silence fade duration.
            try:
                silence_fade = float(self.config.get("silence_fade_duration", 0.6))
            except Exception:
                silence_fade = 0.6
            fade_window = max(0.1, silence_fade)
            frac = max(0.0, min(1.0, dt_silence / fade_window))
            fade_prev = 1.0 - frac


        # --------------------------------------------------------------
        # Layout: positions of lines with optional slide
        # --------------------------------------------------------------
        center_x = rect.center().x()
        center_y = rect.center().y()
        spacing_factor = float(self.config.get("vertical_spacing_factor", 1.1))

        current_width = (
            fm_current.horizontalAdvance(visual_current_line) if visual_current_line else 0
        )
        current_height = fm_current.height()

        x_current = center_x - current_width / 2.0

        # "Static" gap based on current line height and spacing
        static_gap = spacing_factor * float(current_height)

        if is_silence:
            # Silence fade-out: only the "previous" text is shown,
            # sliding from center upwards as it disappears.
            baseline_current = center_y  # not used
            if self._silence_start_t is not None and t is not None:
                dt_silence = max(0.0, t - self._silence_start_t)
                try:
                    silence_fade = float(self.config.get("silence_fade_duration", 0.6))
                except Exception:
                    silence_fade = 0.6
                slide_window = max(0.1, silence_fade)
                frac_slide = max(0.0, min(1.0, dt_silence / slide_window))
            else:
                frac_slide = 0.0

            # Pour le silence, on peut rester sur la géométrie statique :
            gap_for_silence = static_gap
            baseline_prev = center_y - frac_slide * gap_for_silence

        else:
            if not sliding or slide_duration <= 0.0:
                # Static layout: previous line sits above the current one.
                # Reset any frozen slide geometry.
                self._slide_center_y = None
                self._slide_gap = None

                baseline_current = center_y
                baseline_prev = center_y - static_gap
            else:
                # Mirror slide with frozen geometry:
                #   - outgoing line: from center -> center - gap
                #   - incoming line: from center + gap -> center
                # On fige center_y et gap au début du slide pour éviter que
                # des changements de layout ultérieurs ne déplacent la trajectoire.
                if self._slide_center_y is None or self._slide_gap is None:
                    self._slide_center_y = center_y
                    self._slide_gap = static_gap

                cy = float(self._slide_center_y)
                gap = float(self._slide_gap)

                eased = pow(slide_progress, 0.75)
                baseline_current = cy + (1.0 - eased) * gap
                baseline_prev = cy - eased * gap

        # --------------------------------------------------------------
        # Draw previous line (small, faded, decorations reduced)
        # --------------------------------------------------------------
        # We only show a "previous" line if:
        #   - we are in a silence fade (is_silence), or
        #   - we have already had at least one genuine previous phrase.
        can_show_previous = is_silence or self._has_real_previous

        if visual_previous_line and can_show_previous:
            painter.save()
            painter.setFont(prev_font)
            prev_width = fm_prev.horizontalAdvance(visual_previous_line)
            x_prev = center_x - prev_width / 2.0

            alpha_base = float(self.config.get("previous_line_alpha", 0.35))
            alpha_base = max(0.0, min(1.0, alpha_base))
            final_alpha = alpha_base * fade_prev

            if final_alpha > 0.0:
                # Fade EVERYTHING (text + background box + outline + shadow)
                painter.setOpacity(final_alpha)

                # For the previous line, we usually want a lighter style:
                # no background box / outline / shadow to avoid heavy ghosts.
                prev_config = dict(self.config)
                prev_config["text_box_enabled"] = False
                prev_config["text_shadow_enabled"] = False
                prev_config["text_outline_enabled"] = False

                prev_color = QColor(base_color)
                prev_color.setAlphaF(1.0)

                draw_styled_text(
                    painter=painter,
                    x=float(x_prev),
                    y=float(baseline_prev),
                    text=raw_prev_for_frame,
                    config=prev_config,
                    base_font=prev_font,
                    base_color=prev_color,
                )

            painter.restore()

        # --------------------------------------------------------------
        # Highlight the active word on the current line (glow behind text)
        # --------------------------------------------------------------
        painter.save()
        painter.setFont(base_font)

        word_start = getattr(ctx, "word_char_start", None)
        word_end = getattr(ctx, "word_char_end", None)

        have_highlight = (
            visual_current_line
            and isinstance(word_start, int)
            and isinstance(word_end, int)
            and 0 <= word_start < word_end <= len(visual_current_line)
        )

        if have_highlight:
            prefix = visual_current_line[:word_start]
            active = visual_current_line[word_start:word_end]

            prefix_width = fm_current.horizontalAdvance(prefix)
            word_width = max(1, fm_current.horizontalAdvance(active))

            x_word_left = x_current + prefix_width
            highlight_padding_x = float(self.config.get("highlight_padding_x", 8.0))
            highlight_padding_y = float(self.config.get("highlight_padding_y", 4.0))

            highlight_color_str = str(
                self.config.get("highlight_color", "#ffff80") or "#ffff80"
            )
            highlight_color = (
                QColor(highlight_color_str)
                if QColor.isValidColor(highlight_color_str)
                else QColor(255, 255, 128)
            )

            amp = self._smoothed_amp
            glow_strength = float(self.config.get("highlight_glow_strength", 0.9))
            glow_strength = max(0.0, glow_strength)

            # Outer glow (larger, softer)
            outer = QColor(highlight_color)
            outer_alpha = min(1.0, 0.25 + amp * glow_strength * 0.6)
            outer.setAlphaF(outer_alpha)

            # Inner box (tighter, stronger)
            inner = QColor(highlight_color)
            inner_alpha = min(1.0, 0.5 + amp * glow_strength)
            inner.setAlphaF(inner_alpha)

            ascent = fm_current.ascent()
            height = fm_current.height()

            outer_factor = 1.6

            outer_rect = QRectF(
                x_word_left - highlight_padding_x * outer_factor,
                baseline_current - ascent - highlight_padding_y * outer_factor,
                word_width + 2 * highlight_padding_x * outer_factor,
                height + 2 * highlight_padding_y * outer_factor,
            )

            inner_rect = QRectF(
                x_word_left - highlight_padding_x,
                baseline_current - ascent - highlight_padding_y,
                word_width + 2 * highlight_padding_x,
                height + 2 * highlight_padding_y,
            )

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(outer)
            painter.drawRoundedRect(outer_rect, 8.0, 8.0)

            painter.setBrush(inner)
            painter.drawRoundedRect(inner_rect, 8.0, 8.0)

        painter.restore()

        # --------------------------------------------------------------
        # Draw current line (on top of the highlight)
        # --------------------------------------------------------------
        if visual_current_line:
            painter.save()
            painter.setFont(base_font)

            draw_styled_text(
                painter=painter,
                x=float(x_current),
                y=float(baseline_current),
                text=raw_current_for_frame,
                config=self.config,
                base_font=base_font,
                base_color=base_color,
            )
            painter.restore()

        painter.end()
