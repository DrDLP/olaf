from __future__ import annotations

"""
double_line_call_response_mirror_next_v2.py

Double-line "call & response" lyrics visualization with a mirrored reflection.

Based on DoubleLineCallResponseVisualization (stable v5).
Adds a mirrored line below the current phrase (reflection effect).

Important limitation:
The host LyricsFrameContext exposes only the active phrase text. Therefore the
mirrored line is a reflection of the current phrase (not a true "next phrase"
preview).
"""


from typing import Any, Dict, Optional

import re

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


class DoubleLineCallResponseMirrorNextVisualization(BaseLyricsVisualization):
    """
    Previous line on top, current line in the center, active word highlighted.
    """

    plugin_id: str = "double_line_call_response_mirror_next"
    plugin_name: str = "Double-line (call & response + mirror)"
    plugin_description: str = (
        "Shows the previous phrase above the current one with karaoke highlighting, "
        "and adds a mirrored reflection of the current phrase below."
    )
    plugin_author: str = "DrDLP"
    plugin_version: str = "1.0.0"

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
        # Track the last seen playback time so we can reset state on backward seeks.
        self._last_t: Optional[float] = None
        self._current_phrase_index: Optional[int] = None
        self._current_line_text: str = ""
        self._previous_line_text: str = ""
        self._previous_line_set_t: Optional[float] = None  # when previous line was last updated

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
        self.config.setdefault("previous_line_alpha", 0.1)
        self.config.setdefault("vertical_spacing_factor", 1.1)

        self.config.setdefault("highlight_color", "#ffff80")
        self.config.setdefault("highlight_padding_x", 8.0)
        self.config.setdefault("highlight_padding_y", 4.0)
        self.config.setdefault("highlight_glow_strength", 0.9)

        # Karaoke-like alpha per word (sung / active / upcoming)
        self.config.setdefault("sung_word_alpha", 0.35)
        self.config.setdefault("active_word_alpha", 1.0)
        self.config.setdefault("upcoming_word_alpha", 0.15)

        # Duration of the slide when a phrase changes (seconds)
        self.config.setdefault("phrase_slide_duration", 0.1)

        # How much the previous line shrinks *while it fades out* (normal mode).
        # 1.0 = no extra shrink, 0.5 = ends at half the previous-line size.
        self.config.setdefault("fade_shrink_factor", 0.8)

        # Mirrored reflection under the current line (visual effect)
        self.config.setdefault("mirror_enabled", True)
        self.config.setdefault("mirror_gap_px", 26.0)
        self.config.setdefault("mirror_alpha", 0.22)
        self.config.setdefault("mirror_scale", 0.95)
        self.config.setdefault("mirror_fade_in_duration", 0.25)
        self.config.setdefault("mirror_text_box_enabled", False)

        # Maximum time (seconds) the previous line can remain visible
        # when there is no current line (e.g. end of song / long gap).
        self.config.setdefault("previous_line_max_age", 2.0)

        # Reasonable widget size and resizing behaviour
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------
    # Runtime state helpers
    # ------------------------------------------------------------------
    def _reset_runtime_state(self, reset_amp: bool = False) -> None:
        """
        Reset transient animation/cache state.

        This is primarily used when the host seeks backwards (e.g. scrubbing
        the preview slider, or starting an export after a live preview).
        Without this, cached lines could incorrectly appear before the first
        phrase of the song.
        """
        if reset_amp:
            self._smoothed_amp = 0.0

        self._current_phrase_index = None
        self._current_line_text = ""
        self._previous_line_text = ""
        self._previous_line_set_t = None
        self._has_real_previous = False

        self._is_sliding = False
        self._slide_start_t = None
        self._sliding_outgoing_line = ""
        self._sliding_incoming_line = ""
        self._slide_center_y = None
        self._slide_gap = None

        self._silence_active = False
        self._silence_start_t = None
        self._silence_last_line = ""

        self._last_t = None

        # Karaoke: keep already-sung words dimmed during intra-phrase gaps
        # (when the host provides no active word on a given frame).
        self._karaoke_last_text = ""
        self._karaoke_sung_boundary = 0


    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """Expose shared text-style parameters plus effect controls."""
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
                "fade_shrink_factor": PluginParameter(
                    name="fade_shrink_factor",
                    label="Fade shrink factor",
                    type="float",
                    default=0.8,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    description="How much the previous line shrinks while it fades out.",
                ),
                "previous_line_max_age": PluginParameter(
                    name="previous_line_max_age",
                    label="Previous line max age (s)",
                    type="float",
                    default=2.0,
                    minimum=0.0,
                    maximum=10.0,
                    step=0.25,
                    description="Maximum time the previous line stays visible during long gaps.",
                ),
                "highlight_color": PluginParameter(
                    name="highlight_color",
                    label="Highlight color",
                    type="color",
                    default="#ffff80",
                    description="Color used for the active word highlight.",
                ),
                "highlight_padding_x": PluginParameter(
                    name="highlight_padding_x",
                    label="Highlight padding X (px)",
                    type="float",
                    default=8.0,
                    minimum=0.0,
                    maximum=50.0,
                    step=1.0,
                    description="Horizontal padding around the highlighted word.",
                ),
                "highlight_padding_y": PluginParameter(
                    name="highlight_padding_y",
                    label="Highlight padding Y (px)",
                    type="float",
                    default=4.0,
                    minimum=0.0,
                    maximum=50.0,
                    step=1.0,
                    description="Vertical padding around the highlighted word.",
                ),
                "highlight_glow_strength": PluginParameter(
                    name="highlight_glow_strength",
                    label="Highlight glow strength",
                    type="float",
                    default=0.9,
                    minimum=0.0,
                    maximum=3.0,
                    step=0.1,
                    description="How strong the highlight glow reacts to amplitude.",
                ),
                "sung_word_alpha": PluginParameter(
                    name="sung_word_alpha",
                    label="Sung word alpha",
                    type="float",
                    default=0.35,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for words already sung.",
                ),
                "active_word_alpha": PluginParameter(
                    name="active_word_alpha",
                    label="Active word alpha",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for the active word.",
                ),
                "upcoming_word_alpha": PluginParameter(
                    name="upcoming_word_alpha",
                    label="Upcoming word alpha",
                    type="float",
                    default=0.15,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    description="Opacity for upcoming words.",
                ),
                "mirror_enabled": PluginParameter(
                    name="mirror_enabled",
                    label="Mirror enabled",
                    type="bool",
                    default=True,
                    description="If checked, draw a subtle mirrored copy below the current line.",
                ),
                "mirror_gap_px": PluginParameter(
                    name="mirror_gap_px",
                    label="Mirror gap (px)",
                    type="float",
                    default=26.0,
                    minimum=0.0,
                    maximum=200.0,
                    step=1.0,
                    description="Vertical gap between the current line and its mirrored copy.",
                ),
                "mirror_alpha": PluginParameter(
                    name="mirror_alpha",
                    label="Mirror alpha",
                    type="float",
                    default=0.22,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.02,
                    description="Opacity of the mirrored copy.",
                ),
                "mirror_scale": PluginParameter(
                    name="mirror_scale",
                    label="Mirror scale",
                    type="float",
                    default=0.95,
                    minimum=0.2,
                    maximum=2.0,
                    step=0.05,
                    description="Scale applied to the mirrored copy.",
                ),
                "mirror_fade_in_duration": PluginParameter(
                    name="mirror_fade_in_duration",
                    label="Mirror fade-in (s)",
                    type="float",
                    default=0.25,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Fade-in time for the mirror when a new phrase starts.",
                ),
                "mirror_text_box_enabled": PluginParameter(
                    name="mirror_text_box_enabled",
                    label="Mirror uses text box",
                    type="bool",
                    default=False,
                    description="If checked, the mirrored copy can also draw the background box.",
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

        Important: if playback time goes backwards (scrubbing / export start),
        we reset cached state to avoid showing stale lines before the first
        phrase of the song.
        """
        # --- Detect backward seeks --------------------------------------------
        try:
            t_now = float(getattr(ctx, "t", 0.0) or 0.0)
        except Exception:
            t_now = 0.0

        if self._last_t is not None and (t_now + 1e-6) < float(self._last_t):
            # We went back in time: clear cached lines and animations.
            self._reset_runtime_state(reset_amp=False)

        # Record time early so every early-return path stays consistent.
        self._last_t = t_now

        # --- Amplitude smoothing ------------------------------------------------
        target = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.2
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target

        phrase_idx = ctx.phrase_index
        line_now = (ctx.text_full_line or "").strip()

        # --- Gap handling: no text on this frame -------------------------------
        if not line_now:
            # Before the first phrase, we must stay completely empty.
            if self._current_phrase_index is None:
                self._silence_active = False
                self._silence_start_t = None
                self._silence_last_line = ""
                self._previous_line_text = ""
                self._current_line_text = ""
                return

            # Normalize gaps to behave like a regular phrase transition:
            # the last visible line slides/shrinks into the "previous line" slot.
            if not self._silence_active:
                self._silence_active = True
                self._silence_start_t = None
                self._silence_last_line = ""

                outgoing = (self._current_line_text or "").strip()
                if outgoing:
                    self._previous_line_text = outgoing
                    self._has_real_previous = True

                    self._previous_line_set_t = t_now

                    # Animate even if there is no incoming line during the gap.
                    self._sliding_outgoing_line = outgoing
                    self._sliding_incoming_line = ""
                    self._slide_start_t = t_now
                    self._is_sliding = True
                    self._slide_center_y = None
                    self._slide_gap = None

                    # Clear current cache so it won't be drawn as "current".
                    self._current_line_text = ""

            # During gaps, do not update phrase index / lines; only amp is smoothed.
            return

        # We have lyrics again: gap is over.
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
            self._previous_line_set_t = None
            self._is_sliding = False
            self._slide_start_t = None
            self._sliding_outgoing_line = ""
            self._sliding_incoming_line = ""

            self._karaoke_last_text = line_now
            self._karaoke_sung_boundary = 0
            return

        if phrase_idx != self._current_phrase_index:
            # Phrase index changed: prepare slide from old center line to new one.
            outgoing = (self._current_line_text or "").strip()
            incoming = line_now

            self._current_phrase_index = phrase_idx
            self._current_line_text = incoming

            # Karaoke boundary resets on a new phrase.
            self._karaoke_last_text = incoming
            self._karaoke_sung_boundary = 0

            if outgoing:
                # Outgoing line becomes the "previous" one after the slide.
                self._previous_line_text = outgoing
                # From now on we have a genuine previous phrase.
                self._has_real_previous = True

                self._previous_line_set_t = t_now

            # Configure slide animation only if both lines are non-empty.
            if outgoing and incoming:
                self._sliding_outgoing_line = outgoing
                self._sliding_incoming_line = incoming
                self._slide_start_t = t_now
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

        # Easing used for slide motion (and outgoing font scaling).
        eased_slide = pow(slide_progress, 0.75) if sliding else 1.0


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

        # --------------------------------------------------------------
        # Layout: positions of lines with optional slide
        # --------------------------------------------------------------
        # Use the shared anchor parameters (text_pos_x / text_pos_y)
        # so the user can move the whole lyrics block.
        anchor_x, anchor_y = self.get_text_anchor(rect)
        center_x = anchor_x
        center_y = anchor_y
        spacing_factor = float(self.config.get("vertical_spacing_factor", 1.1))

        current_width = (
            fm_current.horizontalAdvance(visual_current_line) if visual_current_line else 0
        )
        current_height = fm_current.height()

        x_current = center_x - current_width / 2.0

        # "Static" gap based on current line height and spacing
        static_gap = spacing_factor * float(current_height)

        if is_silence:
            # Silence (no next line available): keep the last line in place and fade it out.
            # This avoids a "fake slide" when the song ends or when there is a long gap.
            baseline_current = center_y  # not used
            baseline_prev = center_y  # no slide when there is no incoming line

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
                # Freeze center_y and gap at the start of the slide so later layout changes
                # do not affect the trajectory.
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

            # Scale behaviour for the "previous" line:
            # - Phrase slide: outgoing line shrinks from 1.0 -> previous_line_scale.
            # - Silence fade: last sung line shrinks from 1.0 -> previous_line_scale.
            # - Normal fade: previous line shrinks a bit further while fading out.
            prev_scale_for_frame = prev_scale

            def _ease(v: float) -> float:
                v = max(0.0, min(1.0, float(v)))
                return pow(v, 0.75)

            if is_silence:
                # Silence (no incoming line): shrink in place to the configured "previous" scale,
                # using the SAME easing curve as a normal phrase transition (but without sliding).
                if self._silence_start_t is not None and t is not None and slide_duration > 0.0:
                    dt_silence = max(0.0, float(t) - float(self._silence_start_t))
                    silence_progress = max(0.0, min(1.0, dt_silence / float(slide_duration)))
                else:
                    silence_progress = 0.0

                prev_scale_for_frame = 1.0 + (prev_scale - 1.0) * _ease(silence_progress)
            elif sliding:
                prev_scale_for_frame = 1.0 + (prev_scale - 1.0) * float(eased_slide)
            else:
                # Extra shrink while fading (keeps the fade feel "alive" instead of static).
                fade_progress = 1.0 - float(fade_prev)  # 0 -> 1
                fade_shrink = float(self.config.get("fade_shrink_factor", 0.8))
                fade_shrink = max(0.2, min(1.0, fade_shrink))
                prev_scale_for_frame = prev_scale * (1.0 + (fade_shrink - 1.0) * _ease(fade_progress))

            prev_font_frame = base_font
            if abs(prev_scale_for_frame - 1.0) > 1e-6:
                prev_font_frame = type(base_font)(base_font)
                prev_font_frame.setPointSize(int(base_point_size * prev_scale_for_frame))

            fm_prev_frame = QFontMetrics(prev_font_frame)
            painter.setFont(prev_font_frame)

            prev_width = fm_prev_frame.horizontalAdvance(visual_previous_line)
            x_prev = center_x - prev_width / 2.0

            alpha_base = float(self.config.get("previous_line_alpha", 0.35))
            alpha_base = max(0.0, min(1.0, alpha_base))

            # If there is no current line (end-of-song / long gap), do not keep the previous
            # phrase on screen indefinitely: fade it out after a configurable maximum age.
            age_factor = 1.0
            try:
                no_current_line = not bool((raw_current_for_frame or "").strip())
                t_val = float(t) if t is not None else None
                set_t = float(self._previous_line_set_t) if self._previous_line_set_t is not None else None
                max_age = float(self.config.get("previous_line_max_age", 2.0))
                if no_current_line and t_val is not None and set_t is not None and max_age <= 0.0:
                    age_factor = 0.0
                elif no_current_line and t_val is not None and set_t is not None and max_age > 0.0:
                    age = max(0.0, t_val - set_t)
                    tail = max(0.05, min(0.5, max_age * 0.25))
                    start_fade = max(0.0, max_age - tail)
                    if age <= start_fade:
                        age_factor = 1.0
                    elif age < max_age:
                        denom = max(1e-6, max_age - start_fade)
                        age_factor = max(0.0, 1.0 - (age - start_fade) / denom)
                    else:
                        age_factor = 0.0
            except Exception:
                age_factor = 1.0

            final_alpha = alpha_base * fade_prev * age_factor
            if final_alpha > 0.0:
                # Fade EVERYTHING (text + background box + outline + shadow)
                painter.setOpacity(final_alpha)

                prev_color = QColor(base_color)
                prev_color.setAlphaF(1.0)

                # Keep the same styling as the current line (outline, etc.)
                draw_styled_text(
                    painter=painter,
                    x=float(x_prev),
                    y=float(baseline_prev),
                    text=raw_prev_for_frame,
                    config=self.config,
                    base_font=prev_font_frame,
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
        # --------------------------------------------------------------
        # Draw current line (on top of the highlight)
        # --------------------------------------------------------------
        if visual_current_line:
            painter.save()
            painter.setFont(base_font)

            text_to_draw = visual_current_line
            fm = fm_current

            def _clamp01(v: float) -> float:
                return max(0.0, min(1.0, float(v)))

            # Karaoke opacity levels (per token)
            try:
                sung_alpha = _clamp01(float(self.config.get("sung_word_alpha", 0.35)))
            except Exception:
                sung_alpha = 0.35
            try:
                active_alpha = _clamp01(float(self.config.get("active_word_alpha", 1.0)))
            except Exception:
                active_alpha = 1.0
            try:
                upcoming_alpha = _clamp01(float(self.config.get("upcoming_word_alpha", 0.15)))
            except Exception:
                upcoming_alpha = 0.15

            a0 = word_start if isinstance(word_start, int) else None
            a1 = word_end if isinstance(word_end, int) else None
            have_active_span = (
                a0 is not None
                and a1 is not None
                and 0 <= a0 < a1 <= len(text_to_draw)
            )

            # Keep a per-line "sung boundary" so already-sung words stay dimmed
            # even during intra-phrase frames where the host provides no active word.
            if text_to_draw != self._karaoke_last_text:
                self._karaoke_last_text = text_to_draw
                self._karaoke_sung_boundary = 0

            if have_active_span:
                # When a word is active, remember its end as the last known sung boundary.
                self._karaoke_sung_boundary = max(int(self._karaoke_sung_boundary), int(a1))

            sung_boundary = int(self._karaoke_sung_boundary)

            def _draw_karaoke_line(
                p: QPainter,
                x_left: float,
                baseline_y: float,
                text_line: str,
                opacity_scale: float,
                *,
                draw_text_box: bool,
            ) -> None:
                """Draw a line with per-word opacity, including outline and shadow.

                We draw the background text box once (optional), then render each token
                with QPainter opacity so the fill/outline/shadow all fade together.
                """
                opacity_scale = _clamp01(opacity_scale)
                if opacity_scale <= 0.0:
                    return

                # Optional: draw ONLY the background box once for the whole line.
                if draw_text_box and bool(self.config.get("text_box_enabled", False)):
                    box_only_cfg = dict(self.config)
                    box_only_cfg["text_shadow_enabled"] = False
                    box_only_cfg["text_outline_enabled"] = False
                    # Transparent fill -> box only
                    transparent = QColor(base_color)
                    transparent.setAlphaF(0.0)

                    p.save()
                    p.setOpacity(opacity_scale)
                    draw_styled_text(
                        painter=p,
                        x=float(x_left),
                        y=float(baseline_y),
                        text=text_line,
                        config=box_only_cfg,
                        base_font=base_font,
                        base_color=transparent,
                    )
                    p.restore()

                # For tokens, never draw the box again (avoids seams).
                token_cfg = dict(self.config)
                token_cfg["text_box_enabled"] = False

                x_cursor = float(x_left)
                last_end = 0

                for m in re.finditer(r"\S+", text_line):
                    start_i = m.start()
                    end_i = m.end()

                    gap_txt = text_line[last_end:start_i]
                    if gap_txt:
                        x_cursor += fm.horizontalAdvance(gap_txt)

                    token = text_line[start_i:end_i]

                    if not have_active_span:
                        token_alpha = sung_alpha if end_i <= sung_boundary else upcoming_alpha
                    else:
                        if end_i <= int(a0):
                            token_alpha = sung_alpha
                        elif start_i >= int(a1):
                            token_alpha = upcoming_alpha
                        else:
                            token_alpha = active_alpha

                    # Opacity must affect outline and shadow too -> use painter opacity.
                    final_opacity = _clamp01(token_alpha) * opacity_scale
                    if final_opacity > 0.0:
                        p.save()
                        p.setOpacity(final_opacity)

                        solid = QColor(base_color)
                        solid.setAlphaF(1.0)

                        draw_styled_text(
                            painter=p,
                            x=float(x_cursor),
                            y=float(baseline_y),
                            text=token,
                            config=token_cfg,
                            base_font=base_font,
                            base_color=solid,
                        )
                        p.restore()

                    x_cursor += fm.horizontalAdvance(token)
                    last_end = end_i

            # Main line
            _draw_karaoke_line(
                painter,
                float(x_current),
                float(baseline_current),
                text_to_draw,
                1.0,
                draw_text_box=True,
            )

            # Mirrored reflection (under the current line)
            if bool(self.config.get("mirror_enabled", True)):
                try:
                    mirror_gap = float(self.config.get("mirror_gap_px", 26.0))
                except Exception:
                    mirror_gap = 26.0
                try:
                    mirror_alpha = _clamp01(float(self.config.get("mirror_alpha", 0.22)))
                except Exception:
                    mirror_alpha = 0.22
                try:
                    mirror_scale = float(self.config.get("mirror_scale", 0.95))
                except Exception:
                    mirror_scale = 0.95
                mirror_scale = max(0.01, mirror_scale)

                try:
                    fade_in_s = float(self.config.get("mirror_fade_in_duration", 0.25))
                except Exception:
                    fade_in_s = 0.25
                fade_in_s = max(0.0, fade_in_s)

                mirror_fade = 1.0
                if fade_in_s > 1e-6:
                    mirror_fade = _clamp01(local_t / fade_in_s)

                mirror_opacity = mirror_alpha * mirror_fade
                if mirror_opacity > 0.0:
                    painter.save()

                    # Flip vertically around an axis placed *below* the current baseline.
                    axis_y = float(baseline_current) + float(mirror_gap)
                    painter.translate(float(center_x), axis_y)
                    painter.scale(float(mirror_scale), -float(mirror_scale))
                    painter.translate(-float(center_x), -axis_y)

                    _draw_karaoke_line(
                        painter,
                        float(x_current),
                        float(baseline_current),
                        text_to_draw,
                        mirror_opacity,
                        draw_text_box=bool(self.config.get("mirror_text_box_enabled", False)),
                    )

                    painter.restore()

            painter.restore()


        painter.end()
