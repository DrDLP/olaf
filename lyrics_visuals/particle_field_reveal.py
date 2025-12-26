from __future__ import annotations

"""
Particle-field reveal lyrics visualization.

Each phrase (or active word) emerges from a swarm of particles:

  - Particles start far away and converge into the glyph shapes.
  - Only particles are drawn: no solid text.
  - Outline and shadow also use particles.
  - Optional "neon" style adds a soft glow around each point.

Layout options:
  - Horizontal full phrase.
  - Vertical (one word per line, centered).
  - Word-by-word mode: only the active word is rendered.

The background behaviour is the same as in other lyrics plugins:
it uses the project cover when available, or a gradient / solid color.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random
import math
import hashlib

# Optional: NumPy accelerates mask-based sampling ("Pro" option).
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter,
    QColor,
    QFontMetrics,
    QLinearGradient,
    QBrush,
    QImage,
    QPolygonF,
    QPen,
    QPainterPath,
    QPainterPathStroker,
)
from PyQt6.QtWidgets import QWidget, QSizePolicy

from olaf_app.lyrics_visualization_api import BaseLyricsVisualization, LyricsFrameContext
from olaf_app.visualization_api import PluginParameter
from olaf_app.lyrics_text_style import (
    apply_default_text_style_config,
    text_style_parameters,
    build_qfont_from_config,
    font_color_from_config,
)


# ---------------------------------------------------------------------------
# Particle data structure
# ---------------------------------------------------------------------------


@dataclass
class Particle:
    """
    Single particle used to reveal the text.

    Positions are in widget coordinates.
    """

    sx: float  # start x
    sy: float  # start y
    tx: float  # target x
    ty: float  # target y
    start_t: float  # global time when this particle starts moving
    duration: float  # duration of the motion in seconds
    size: float  # base radius
    phase: float  # small random phase used for subtle jitter
    base_color: QColor  # base color of this particle (alpha is modulated at draw time)


# ---------------------------------------------------------------------------
# Visualization plugin
# ---------------------------------------------------------------------------


class ParticleFieldRevealVisualization(BaseLyricsVisualization):
    """
    Make the current phrase (or active word) emerge from a field of particles
    that literally form the glyphs, outline, and shadow of the text.
    No solid text is drawn: only particles are visible.
    """

    plugin_id: str = "particle_field_reveal"
    plugin_name: str = "Particle field reveal"
    plugin_description: str = (
        "The phrase or active word emerges from a dense swarm of particles that "
        "converge into letter, outline, and shadow shapes. Only particles are "
        "visible; there is no solid text."
    )
    plugin_author: str = "DrDLP"
    plugin_version: str = "2.5.0"

    # ------------------------------------------------------------------ #
    # Construction / defaults                                            #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(config=config, parent=parent)

        # Ensure all shared text-style keys exist in the config.
        apply_default_text_style_config(self.config)

        # Default background: use project cover.
        self.config.setdefault("background_mode", "cover")

        # Shared text anchor (host may expose these sliders globally).
        self.config.setdefault("text_pos_x", 0.5)
        self.config.setdefault("text_pos_y", 0.5)

        # Spacing between lines when the input text contains newlines, or in vertical layout.
        self.config.setdefault("phrase_spacing_px", 10)

        # Particle spawning anisotropy ("variation X/Y").
        self.config.setdefault("spawn_spread_x", 1.0)
        self.config.setdefault("spawn_spread_y", 1.0)

        # Per-axis jitter scaling during the reveal.
        self.config.setdefault("jitter_x_scale", 1.0)
        self.config.setdefault("jitter_y_scale", 1.0)

        # Rendering backend / quality presets.
        self.config.setdefault("render_backend", "points_fast")
        self.config.setdefault("antialiasing", False)
        self.config.setdefault("alpha_quantization_steps", 12)

        # Keep particle counts under control on large canvases (especially 4K export).
        self.config.setdefault("auto_cap_enabled", True)
        self.config.setdefault("max_particles_per_mp", 30000)

        # Deterministic seed (stable across frames / exports).
        self.config.setdefault("random_seed", 1337)

        # "Pro" sampling: use a raster mask + NumPy when available.
        self.config.setdefault("sampling_mode", "contains")
        self.config.setdefault("mask_sampling_max_px", 2_000_000)

        # Audio / phrase state
        self._smoothed_amp: float = 0.0
        self._current_phrase_index: Optional[int] = None
        self._current_line_text: str = ""

        # Timing for the current unit (phrase or word).
        self._phrase_start_t: Optional[float] = None

        # Silence fade-out state
        self._silence_active: bool = False
        self._silence_start_t: Optional[float] = None

        # Particle system for the current unit
        self._particles: List[Particle] = []
        self._phrase_layout_ready: bool = False

        # Cached text layout (not drawn, but kept for diagnostics)
        self._text_x: float = 0.0
        self._text_baseline_y: float = 0.0
        self._text_width: float = 0.0
        self._text_height: float = 0.0

        # Cached painter path for glyphs (in widget coordinates)
        self._text_path: Optional[QPainterPath] = None
        self._layout_anchor_x: float = 0.0
        self._layout_anchor_y: float = 0.0

        # Reasonable widget size and resizing behaviour
        self.setMinimumHeight(160)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    # ------------------------------------------------------------------ #
    # Parameters                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Expose shared text-style parameters + plugin-specific controls.
        """
        params = dict(text_style_parameters())

        params.update(
            {
                # Layout: horizontal phrase or vertical words
                "layout_mode": PluginParameter(
                    name="layout_mode",
                    label="Layout mode",
                    type="enum",
                    default="horizontal",
                    choices=["horizontal", "vertical_words"],
                    description=(
                        "Horizontal phrase on one line, or vertically stacked words "
                        "(one word per line, centered)."
                    ),
                ),
                # Word-by-word option: only the active word is rendered
                "word_by_word": PluginParameter(
                    name="word_by_word",
                    label="Word-by-word (active word only)",
                    type="bool",
                    default=False,
                    description=(
                        "If enabled, only the active word is formed by particles "
                        "instead of the full phrase."
                    ),
                ),
                # Number of particles (real value)
                "particle_target_count": PluginParameter(
                    name="particle_target_count",
                    label="Approx. particle count",
                    type="int",
                    default=20000,
                    minimum=2000,
                    maximum=5000000,
                    step=2000,
                    description=(
                        "Approximate number of particles used to form each phrase/word. "
                        "Higher values produce more solid letter shapes but cost more CPU."
                    ),
                ),
                "particle_reveal_duration": PluginParameter(
                    name="particle_reveal_duration",
                    label="Reveal duration (s)",
                    type="float",
                    default=1.0,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.05,
                    description=(
                        "Duration of the convergence of particles towards the glyphs. "
                        "The animation starts this many seconds BEFORE the phrase/word "
                        "is sung, so it is fully formed on time."
                    ),
                ),
                "particle_spawn_spread": PluginParameter(
                    name="particle_spawn_spread",
                    label="Spawn spread (s)",
                    type="float",
                    default=0.5,
                    minimum=0.0,
                    maximum=3.0,
                    step=0.05,
                    description=(
                        "Temporal spread (jitter) of particle start times. "
                        "Higher values create a more staggered reveal."
                    ),
                ),
                "particle_size_min": PluginParameter(
                    name="particle_size_min",
                    label="Particle size min",
                    type="float",
                    default=0.3,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    description="Minimum radius of particles (in pixels).",
                ),
                "particle_size_max": PluginParameter(
                    name="particle_size_max",
                    label="Particle size max",
                    type="float",
                    default=0.7,
                    minimum=0.2,
                    maximum=2.0,
                    step=0.05,
                    description="Maximum radius of particles (in pixels).",
                ),
                "particle_field_spread": PluginParameter(
                    name="particle_field_spread",
                    label="Field spread factor",
                    type="float",
                    default=1.2,
                    minimum=0.2,
                    maximum=3.0,
                    step=0.1,
                    description=(
                        "How far from the text area particles initially spawn "
                        "(in units of min(canvas_width, canvas_height))."
                    ),
                ),
                "particle_trail_opacity": PluginParameter(
                    name="particle_trail_opacity",
                    label="Particle opacity",
                    type="float",
                    default=0.9,
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    description="Base opacity of particles at the end of the reveal.",
                ),
                # Rendering / performance
                "render_backend": PluginParameter(
                    name="render_backend",
                    label="Render backend",
                    type="enum",
                    default="points_fast",
                    choices=["points_fast", "points_pro", "ellipses"],
                    description=(
                        "How particles are drawn. 'points_fast' batches draw calls and is much faster. "
                        "'points_pro' adds stronger batching (alpha/size quantization) for very large particle counts. "
                        "'ellipses' keeps the legacy look but is the slowest."
                    ),
                ),
                "antialiasing": PluginParameter(
                    name="antialiasing",
                    label="Antialiasing",
                    type="bool",
                    default=False,
                    description="Enable antialiasing (prettier, slower).",
                ),
                "alpha_quantization_steps": PluginParameter(
                    name="alpha_quantization_steps",
                    label="Alpha quantization steps",
                    type="int",
                    default=12,
                    minimum=4,
                    maximum=32,
                    step=1,
                    description=("Number of discrete alpha levels used by the batched point renderer."),
                ),
                "auto_cap_enabled": PluginParameter(
                    name="auto_cap_enabled",
                    label="Auto cap particles",
                    type="bool",
                    default=True,
                    description="Cap particle counts based on canvas size to avoid 4K slowdowns.",
                ),
                "max_particles_per_mp": PluginParameter(
                    name="max_particles_per_mp",
                    label="Max particles / MP",
                    type="int",
                    default=20000,
                    minimum=2000,
                    maximum=60000,
                    step=1000,
                    description="Particle cap in particles per megapixel when auto-cap is enabled.",
                ),

                # Layout
                "phrase_spacing_px": PluginParameter(
                    name="phrase_spacing_px",
                    label="Phrase/line spacing (px)",
                    type="int",
                    default=10,
                    minimum=-300,
                    maximum=200,
                    step=2,
                    description="Extra spacing between lines/newlines or between stacked words in vertical mode.",
                ),

                # Spawn / motion (variation X/Y)
                "spawn_spread_x": PluginParameter(
                    name="spawn_spread_x",
                    label="Spawn spread X",
                    type="float",
                    default=1.0,
                    minimum=0.2,
                    maximum=3.0,
                    step=0.05,
                    description="Multiply initial spawn spread on X (elliptic field).",
                ),
                "spawn_spread_y": PluginParameter(
                    name="spawn_spread_y",
                    label="Spawn spread Y",
                    type="float",
                    default=1.0,
                    minimum=0.2,
                    maximum=3.0,
                    step=0.05,
                    description="Multiply initial spawn spread on Y (elliptic field).",
                ),
                "jitter_x_scale": PluginParameter(
                    name="jitter_x_scale",
                    label="Jitter scale X",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Scale the reveal jitter on X.",
                ),
                "jitter_y_scale": PluginParameter(
                    name="jitter_y_scale",
                    label="Jitter scale Y",
                    type="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    description="Scale the reveal jitter on Y.",
                ),

                # Determinism / sampling
                "random_seed": PluginParameter(
                    name="random_seed",
                    label="Random seed",
                    type="int",
                    default=1337,
                    minimum=0,
                    maximum=1_000_000,
                    step=1,
                    description="Base seed for deterministic particle layouts.",
                ),
                "sampling_mode": PluginParameter(
                    name="sampling_mode",
                    label="Sampling mode",
                    type="enum",
                    default="contains",
                    choices=["contains", "mask_numpy"],
                    description=("Point sampling method. 'mask_numpy' uses a raster mask (requires NumPy)."),
                ),
                "mask_sampling_max_px": PluginParameter(
                    name="mask_sampling_max_px",
                    label="Mask sampling max pixels",
                    type="int",
                    default=2_000_000,
                    minimum=200_000,
                    maximum=10_000_000,
                    step=100_000,
                    description="Safety limit for mask sampling (width*height). Above this, falls back to contains().",
                ),

                "style_mode": PluginParameter(
                    name="style_mode",
                    label="Style mode",
                    type="enum",
                    default="classic",
                    choices=["classic", "neon"],
                    description="Classic look or futuristic neon glow for particles.",
                ),
            }
        )

        return params

    # ------------------------------------------------------------------ #
    # Frame update logic                                                 #
    # ------------------------------------------------------------------ #

    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Update internal animation state from the current frame context.
        """
        # Smooth amplitude for background use.
        target_amp = max(0.0, min(1.0, float(ctx.amp)))
        alpha = 0.2
        self._smoothed_amp = (1.0 - alpha) * self._smoothed_amp + alpha * target_amp

        t = float(getattr(ctx, "t", 0.0) or 0.0)

        # Determine if we operate on the full phrase or on the active word.
        word_by_word = bool(self.config.get("word_by_word", False))

        if word_by_word:
            unit_index = ctx.word_index
            unit_text = (ctx.text_active_word or "").strip() if ctx.text_active_word else ""
        else:
            unit_index = ctx.phrase_index
            unit_text = (ctx.text_full_line or "").strip()

        # Silence / no active unit: enable fade-out and stop updating particles.
        if unit_index is None or not unit_text:
            if not self._silence_active:
                self._silence_active = True
                self._silence_start_t = t
            return
        else:
            self._silence_active = False
            self._silence_start_t = None

        # Reveal duration used for pre-reveal.
        try:
            reveal_duration = float(self.config.get("particle_reveal_duration", 1.0))
        except Exception:
            reveal_duration = 1.0
        if reveal_duration < 0.05:
            reveal_duration = 0.05

        # First unit (phrase or word)
        if self._current_phrase_index is None:
            self._current_phrase_index = unit_index
            self._current_line_text = unit_text
            # Start animation earlier so text is ready when sung.
            self._phrase_start_t = t - reveal_duration
            self._phrase_layout_ready = False
            self._text_path = None
            self._particles.clear()
            return

        # New unit (phrase index changed or text changed)
        if unit_index != self._current_phrase_index or unit_text != self._current_line_text:
            self._current_phrase_index = unit_index
            self._current_line_text = unit_text
            self._phrase_start_t = t - reveal_duration
            self._phrase_layout_ready = False
            self._text_path = None
            self._particles.clear()

    # ------------------------------------------------------------------ #
    # Background painting (shared behaviour)                             #
    # ------------------------------------------------------------------ #

    def _paint_background(self, painter: QPainter, rect) -> None:
        """
        Fill the global background (behind the particles).
        """
        mode = str(self.config.get("background_mode", "gradient") or "gradient")

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
        elif mode == "solid":
            col_str = str(self.config.get("background_color", "#000000") or "#000000")
            color = QColor(col_str) if QColor.isValidColor(col_str) else QColor(0, 0, 0)
            painter.fillRect(rect, color)
        else:
            # Gradient background, slightly boosted by audio.
            top_str = str(self.config.get("background_gradient_top", "#101010") or "#101010")
            bottom_str = str(
                self.config.get("background_gradient_bottom", "#402840") or "#402840"
            )
            top_color = (
                QColor(top_str) if QColor.isValidColor(top_str) else QColor(16, 16, 16)
            )
            bottom_color = (
                QColor(bottom_str)
                if QColor.isValidColor(bottom_str)
                else QColor(64, 40, 64)
            )

            # Slight brightening based on amplitude
            bottom_color = QColor(
                min(255, bottom_color.red() + int(60 * amp)),
                bottom_color.green(),
                min(255, bottom_color.blue() + int(60 * amp)),
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

    # ------------------------------------------------------------------ #
    # Utility: color helpers                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _color_from_config(config: Dict[str, Any], key: str, default_hex: str) -> QColor:
        """
        Return a QColor from config[key], falling back to *default_hex*
        when missing or invalid.
        """
        raw = str(config.get(key) or default_hex)
        color = QColor(raw)
        if not color.isValid():
            color = QColor(default_hex)
        return color

    # ------------------------------------------------------------------ #
    # Utility: shared anchor + deterministic seed                        #
    # ------------------------------------------------------------------ #

    def _get_text_anchor_xy(self, rect) -> tuple[float, float]:
        """Return the current text anchor in widget coordinates."""
        getter = getattr(self, "get_text_anchor", None)
        if callable(getter):
            try:
                x, y = getter(rect)
                return float(x), float(y)
            except Exception:
                pass

        try:
            nx = float(self.config.get("text_pos_x", 0.5))
        except Exception:
            nx = 0.5
        try:
            ny = float(self.config.get("text_pos_y", 0.5))
        except Exception:
            ny = 0.5
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        return float(rect.left() + rect.width() * nx), float(rect.top() + rect.height() * ny)

    def _compute_unit_seed(self, unit_text: str, unit_index: Optional[int]) -> int:
        """Return a stable 32-bit seed for the current phrase/word."""
        base_seed = int(self.config.get("random_seed", 1337) or 1337)
        key = f"{unit_index}|{unit_text}|{bool(self.config.get('word_by_word', False))}"
        digest = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()[:8]
        h = int(digest, 16)
        return int((base_seed ^ h) & 0xFFFFFFFF)


    # ------------------------------------------------------------------ #
    # Particles generation                                               #
    # ------------------------------------------------------------------ #

    def _ensure_phrase_layout_and_particles(
        self,
        painter: QPainter,
        fm: QFontMetrics,
        rect,
        t: float,
    ) -> None:
        """
        Compute text layout and generate particles if needed.

        Particle targets are sampled INSIDE the glyph path, then reused
        with small offsets to build outline and shadow layers.
        """
        if self._phrase_layout_ready or not self._current_line_text:
            return

        text = self._current_line_text
        if bool(self.config.get("capitalize_all", False)):
            text = text.upper()

        layout_mode = str(self.config.get("layout_mode", "horizontal") or "horizontal")

        # Extra spacing between lines (newlines) or between stacked words in vertical mode.
        try:
            phrase_spacing_px = float(self.config.get("phrase_spacing_px", 10))
        except Exception:
            phrase_spacing_px = 10.0
        # Allow negative spacing to tighten lines, but keep a minimal readable step.
        # Negative values are useful for compact layouts on small outputs.
        # We still clamp the final line step to avoid complete overlap.
        # (phrase_spacing_px can be negative via the UI slider.)


        path = QPainterPath()

        # Build a painter path for the glyphs according to layout mode:
        #   - "horizontal": full phrase, supports multi-line text via \n
        #   - "vertical_words": one word per line (stacked)
        if layout_mode == "vertical_words":
            units = [w for w in text.split(" ") if w]
            if not units:
                units = [text]
        else:
            units = text.splitlines()
            if not units:
                units = [text]

        base_step = float(fm.height()) * 1.15
        min_step = float(fm.height()) * 0.30  # safety clamp against extreme overlaps
        line_step = max(min_step, base_step + phrase_spacing_px)
        for i, line in enumerate(units):
            if not line:
                continue
            w = fm.horizontalAdvance(line)
            x = -0.5 * float(w)
            baseline_y = float(i) * line_step
            path.addText(x, baseline_y, painter.font(), line)

        local_br: QRectF = path.boundingRect()
        if local_br.width() <= 0.0 or local_br.height() <= 0.0:
            self._text_path = None
            self._particles = []
            self._phrase_layout_ready = True
            return

        # Place the glyph group on the shared anchor (text_pos_x/text_pos_y).
        anchor_x, anchor_y = self._get_text_anchor_xy(rect)
        dx = anchor_x - local_br.center().x()
        dy = anchor_y - local_br.center().y()
        path.translate(dx, dy)
        self._layout_anchor_x = float(anchor_x)
        self._layout_anchor_y = float(anchor_y)

        global_br: QRectF = path.boundingRect()
        self._text_x = global_br.left()
        self._text_baseline_y = global_br.bottom() - fm.descent()
        self._text_width = float(global_br.width())
        self._text_height = float(global_br.height())
        self._text_path = path

        # Particle count from UI (real value) + optional auto-cap based on canvas size.
        try:
            target_count = int(self.config.get("particle_target_count", 20000))
        except Exception:
            target_count = 20000

        if bool(self.config.get("auto_cap_enabled", True)):
            try:
                max_ppmp = int(self.config.get("max_particles_per_mp", 20000))
            except Exception:
                max_ppmp = 20000
            max_ppmp = max(2000, min(60000, max_ppmp))
            mp = (max(1, rect.width()) * max(1, rect.height())) / 1_000_000.0
            auto_cap = int(max_ppmp * mp)
            target_count = min(target_count, auto_cap)

        target_count = max(2000, min(90000, int(target_count)))

        # Animation parameters
        try:
            reveal_duration = float(self.config.get("particle_reveal_duration", 1.0))
        except Exception:
            reveal_duration = 1.0
        reveal_duration = max(0.05, reveal_duration)

        try:
            spawn_spread = float(self.config.get("particle_spawn_spread", 0.5))
        except Exception:
            spawn_spread = 0.5
        spawn_spread = max(0.0, spawn_spread)

        try:
            size_min = float(self.config.get("particle_size_min", 0.3))
        except Exception:
            size_min = 0.3
        try:
            size_max = float(self.config.get("particle_size_max", 0.7))
        except Exception:
            size_max = 0.7
        if size_max < size_min:
            size_max = size_min

        canvas_w = rect.width()
        canvas_h = rect.height()
        min_dim = float(max(1, min(canvas_w, canvas_h)))
        try:
            spread_factor = float(self.config.get("particle_field_spread", 1.2))
        except Exception:
            spread_factor = 1.2
        spread_radius = spread_factor * min_dim

        rng = random.Random()
        rng.seed(self._compute_unit_seed(text, self._current_phrase_index))

        cx = global_br.center().x()
        cy = global_br.center().y()

        # Spawn anisotropy (variation X/Y).
        try:
            spread_x = float(self.config.get("spawn_spread_x", 1.0))
        except Exception:
            spread_x = 1.0
        try:
            spread_y = float(self.config.get("spawn_spread_y", 1.0))
        except Exception:
            spread_y = 1.0
        spread_x = max(0.05, spread_x)
        spread_y = max(0.05, spread_y)

        # Colors from text style config
        main_color = font_color_from_config(self.config)
        outline_color = self._color_from_config(self.config, "text_outline_color", "#000000")
        shadow_color = self._color_from_config(self.config, "text_shadow_color", "#000000")

        outline_enabled = bool(self.config.get("text_outline_enabled", True))
        shadow_enabled = bool(self.config.get("text_shadow_enabled", True))

        # Budget main / outline / shadow
        main_count = int(target_count * 0.6)
        outline_count = int(target_count * 0.25) if outline_enabled else 0
        shadow_count = target_count - main_count - outline_count if shadow_enabled else 0
        if shadow_count < 0:
            shadow_count = 0

        particles: List[Particle] = []

        phrase_start = self._phrase_start_t or t

        def make_particle(tx: float, ty: float, color: QColor) -> Particle:
            """
            Build a Particle with random spawn location and timing
            for the given target point and base color.
            """
            angle = rng.random() * 2.0 * 3.14159265
            radius = rng.random() * spread_radius
            sx = cx + radius * math.cos(angle) * spread_x
            sy = cy + radius * math.sin(angle) * spread_y

            jitter = (rng.random() * 2.0 - 1.0) * spawn_spread * 0.5
            p_start = phrase_start + max(0.0, jitter)
            p_duration = reveal_duration * (0.7 + 0.6 * rng.random())

            size = size_min + (size_max - size_min) * (rng.random() ** 0.7)
            phase = rng.random() * 6.2831853

            return Particle(
                sx=float(sx),
                sy=float(sy),
                tx=float(tx),
                ty=float(ty),
                start_t=float(p_start),
                duration=float(max(0.05, p_duration)),
                size=float(size),
                phase=float(phase),
                base_color=color,
            )

        # Sample base points inside glyph path (for main layer)
        def sample_inside_path(
            target_path: QPainterPath, br: QRectF, count: int
        ) -> List[QPointF]:
            if count <= 0:
                return []

            # "Pro" option: rasterize the path and sample pixels using NumPy (when available).
            sampling_mode = str(self.config.get("sampling_mode", "contains") or "contains")
            if sampling_mode == "mask_numpy" and np is not None:
                try:
                    max_px = int(self.config.get("mask_sampling_max_px", 2_000_000))
                except Exception:
                    max_px = 2_000_000

                w = max(1, int(br.width()))
                h = max(1, int(br.height()))
                if w * h <= max_px:
                    try:
                        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
                        img.fill(0)
                        qp = QPainter(img)
                        qp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                        qp.translate(-br.left(), -br.top())
                        qp.fillPath(target_path, QColor(255, 255, 255))
                        qp.end()

                        buf = img.bits()
                        buf.setsize(img.sizeInBytes())
                        bpl = img.bytesPerLine()
                        raw = np.frombuffer(buf, dtype=np.uint8).reshape((h, bpl))
                        alpha = raw[:, 3::4][:, :w]
                        ys, xs = np.nonzero(alpha > 0)
                        n = int(xs.size)
                        if n > 0:
                            pts: List[QPointF] = []
                            for _ in range(count):
                                j = rng.randrange(n)
                                pts.append(QPointF(br.left() + float(xs[j]) + 0.5, br.top() + float(ys[j]) + 0.5))
                            return pts
                    except Exception:
                        pass

            # Default sampling: random points tested with QPainterPath.contains().
            pts: List[QPointF] = []
            max_attempts = count * 30
            attempts = 0
            while len(pts) < count and attempts < max_attempts:
                attempts += 1
                rx = br.left() + rng.random() * br.width()
                ry = br.top() + rng.random() * br.height()
                p = QPointF(rx, ry)
                if target_path.contains(p):
                    pts.append(p)
            while len(pts) < count:
                rx = br.left() + rng.random() * br.width()
                ry = br.top() + rng.random() * br.height()
                pts.append(QPointF(rx, ry))
            return pts

        # Sample points along the glyph outline path
        def sample_outline_ring(
            target_path: QPainterPath,
            br: QRectF,
            count: int,
            outline_width: float,
        ) -> List[QPointF]:
            """
            Sample points along the glyph outline by creating a stroked path
            and reusing the interior sampler on that stroke.

            This keeps outline particles on a thin ring instead of filling
            the whole letter.
            """
            stroker = QPainterPathStroker()
            # Stroke width ≈ outline width (tuned visually)
            stroker.setWidth(max(0.5, float(outline_width)))
            stroker.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            stroker.setCapStyle(Qt.PenCapStyle.RoundCap)

            stroke_path = stroker.createStroke(target_path)
            return sample_inside_path(stroke_path, br, count)


        base_points = sample_inside_path(path, global_br, max(main_count, 1))

        # Control drawing order via list order:
        #   shadow -> text -> outline

        # --- Layer 1: shadow (drawn at the bottom) --------------------------
        if shadow_enabled and shadow_count > 0:
            offset_x = int(self.config.get("text_shadow_offset_x", 3))
            offset_y = int(self.config.get("text_shadow_offset_y", 3))

            for i in range(shadow_count):
                p_idx = i % len(base_points)
                base_p = base_points[p_idx]
                tx = float(base_p.x() + offset_x)
                ty = float(base_p.y() + offset_y)
                particles.append(make_particle(tx, ty, shadow_color))

        # --- Layer 2: main fill (glyph interior) ----------------------------
        if main_count > 0:
            for i in range(main_count):
                p_idx = i % len(base_points)
                p = base_points[p_idx]
                particles.append(
                    make_particle(float(p.x()), float(p.y()), main_color)
                )

        # --- Layer 3: outline (tight ring around glyph edge, on top) --------
        if outline_enabled and outline_count > 0:
            width_px = float(self.config.get("text_outline_width", 2.0))
            width_px = max(0.5, width_px)

            # Sample points specifically on the outline stroke, not inside fill
            outline_points = sample_outline_ring(
                path,
                global_br,
                max(outline_count, 1),
                width_px,
            )

            # Small radial jitter around the stroke so the ring does not look too rigid
            outline_radius = width_px * 0.3

            for i in range(outline_count):
                p_idx = i % len(outline_points)
                base_p = outline_points[p_idx]

                angle = rng.random() * 2.0 * 3.14159265
                dist = outline_radius * (rng.random() - 0.5)
                tx = float(base_p.x() + dist * math.cos(angle))
                ty = float(base_p.y() + dist * math.sin(angle))

                particles.append(make_particle(tx, ty, outline_color))

        self._particles = particles
        self._phrase_layout_ready = True

    # ------------------------------------------------------------------ #
    # Painting                                                           #
    # ------------------------------------------------------------------ #


    # ------------------------------------------------------------------ #
    # Rendering backends                                                 #
    # ------------------------------------------------------------------ #

    def _draw_particles_ellipses(
        self,
        painter: QPainter,
        t: float,
        reveal_eased: float,
        silence_mult: float,
        is_neon: bool,
        trail_opacity: float,
        offset_x: float,
        offset_y: float,
    ) -> None:
        """Legacy renderer: one drawEllipse() (or two in neon) per particle."""
        try:
            jx = float(self.config.get("jitter_x_scale", 1.0))
        except Exception:
            jx = 1.0
        try:
            jy = float(self.config.get("jitter_y_scale", 1.0))
        except Exception:
            jy = 1.0

        painter.setPen(QPen(Qt.PenStyle.NoPen))

        for p in self._particles:
            local_t = t - p.start_t
            if local_t <= 0.0:
                progress = 0.0
            elif local_t >= p.duration:
                progress = 1.0
            else:
                progress = max(0.0, min(1.0, local_t / p.duration))

            eased = 1.0 - pow(1.0 - progress, 3.0)

            x = p.sx + (p.tx - p.sx) * eased + offset_x
            y = p.sy + (p.ty - p.sy) * eased + offset_y

            jitter_amp = 0.8 * (1.0 - eased)
            x += jitter_amp * math.sin(p.phase + 4.0 * eased) * jx
            y += jitter_amp * math.cos(p.phase + 3.0 * eased) * jy

            size = p.size

            base_alpha = trail_opacity * (0.3 + 0.7 * progress) * reveal_eased
            final_alpha = base_alpha * silence_mult
            if final_alpha <= 0.01:
                continue

            if is_neon:
                glow = QColor(p.base_color)
                glow.setAlphaF(max(0.0, min(1.0, final_alpha * 0.5)))
                painter.setBrush(glow)
                glow_r = size * 2.4
                painter.drawEllipse(QRectF(x - glow_r, y - glow_r, 2.0 * glow_r, 2.0 * glow_r))

            c = QColor(p.base_color)
            c.setAlphaF(max(0.0, min(1.0, final_alpha)))
            painter.setBrush(c)
            r = size
            painter.drawEllipse(QRectF(x - r, y - r, 2.0 * r, 2.0 * r))

    def _draw_particles_points(
        self,
        painter: QPainter,
        t: float,
        reveal_eased: float,
        silence_mult: float,
        is_neon: bool,
        trail_opacity: float,
        offset_x: float,
        offset_y: float,
        pro: bool = False,
    ) -> None:
        """Fast renderer: batches draw calls via drawPoints()."""
        try:
            alpha_steps = int(self.config.get("alpha_quantization_steps", 12))
        except Exception:
            alpha_steps = 12
        alpha_steps = max(4, min(32, alpha_steps))

        try:
            jx = float(self.config.get("jitter_x_scale", 1.0))
        except Exception:
            jx = 1.0
        try:
            jy = float(self.config.get("jitter_y_scale", 1.0))
        except Exception:
            jy = 1.0

        size_bins = 6 if pro else 3
        core_groups: dict[tuple[int, int, int, int, int], list[QPointF]] = {}
        glow_groups: dict[tuple[int, int, int, int, int], list[QPointF]] = {}

        if self._particles:
            smin = min(p.size for p in self._particles)
            smax = max(p.size for p in self._particles)
        else:
            smin, smax = 0.3, 0.7
        srange = max(1e-6, float(smax - smin))

        for p in self._particles:
            local_t = t - p.start_t
            if local_t <= 0.0:
                progress = 0.0
            elif local_t >= p.duration:
                progress = 1.0
            else:
                progress = max(0.0, min(1.0, local_t / p.duration))

            eased = 1.0 - pow(1.0 - progress, 3.0)

            x = p.sx + (p.tx - p.sx) * eased + offset_x
            y = p.sy + (p.ty - p.sy) * eased + offset_y

            jitter_amp = 0.8 * (1.0 - eased)
            x += jitter_amp * math.sin(p.phase + 4.0 * eased) * jx
            y += jitter_amp * math.cos(p.phase + 3.0 * eased) * jy

            base_alpha = trail_opacity * (0.3 + 0.7 * progress) * reveal_eased
            final_alpha = base_alpha * silence_mult
            if final_alpha <= 0.01:
                continue

            a_bucket = int(max(0, min(alpha_steps - 1, round(final_alpha * (alpha_steps - 1)))))
            s_bucket = int(max(0, min(size_bins - 1, ((p.size - smin) / srange) * (size_bins - 1))))

            c = QColor(p.base_color)
            key = (c.red(), c.green(), c.blue(), a_bucket, s_bucket)
            core_groups.setdefault(key, []).append(QPointF(float(x), float(y)))

            if is_neon:
                glow_alpha = min(1.0, (a_bucket / max(1, alpha_steps - 1)) * 0.5)
                ga_bucket = int(max(0, min(alpha_steps - 1, round(glow_alpha * (alpha_steps - 1)))))
                gkey = (c.red(), c.green(), c.blue(), ga_bucket, s_bucket)
                glow_groups.setdefault(gkey, []).append(QPointF(float(x), float(y)))

        def bucket_to_alpha(b: int) -> float:
            return max(0.0, min(1.0, b / float(max(1, alpha_steps - 1))))

        def bucket_to_width(b: int, scale: float) -> float:
            if size_bins <= 1:
                s = (smin + smax) * 0.5
            else:
                frac = b / float(size_bins - 1)
                s = smin + frac * srange
            return max(0.75, float(s) * 2.0 * scale)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen()
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        def draw_groups(groups: dict[tuple[int, int, int, int, int], list[QPointF]], scale: float) -> None:
            for (r, g, b, a_b, s_b), pts in groups.items():
                if not pts:
                    continue
                color = QColor(r, g, b)
                color.setAlphaF(bucket_to_alpha(a_b))
                pen.setColor(color)
                pen.setWidthF(bucket_to_width(s_b, scale))
                painter.setPen(pen)
                chunk = 20000 if pro else 50000
                for start in range(0, len(pts), chunk):
                    painter.drawPoints(QPolygonF(pts[start : start + chunk]))

        if is_neon and glow_groups:
            draw_groups(glow_groups, scale=2.4)
        if core_groups:
            draw_groups(core_groups, scale=1.0)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        """
        Render:
          - background cover / gradient,
          - particle field converging towards the glyphs, outline, and shadow,
          - optional fade-out during silences.

        The text itself is NOT drawn; only particles form the letters.
        """
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Background
        self._paint_background(painter, rect)

        ctx = self.current_context()
        if ctx is None:
            painter.end()
            return

        # Typography (only for layout)
        try:
            base_point_size = int(self.config.get("font_size", 40))
        except Exception:
            base_point_size = 40
        base_point_size = max(8, base_point_size)

        base_font = build_qfont_from_config(self.config, painter, base_point_size)
        painter.setFont(base_font)
        fm_current = QFontMetrics(base_font)

        text = self._current_line_text
        if bool(self.config.get("capitalize_all", False)):
            text = text.upper()

        if not text:
            painter.end()
            return

        t = float(getattr(ctx, "t", 0.0) or 0.0)

        # Layout + particles if needed
        self._ensure_phrase_layout_and_particles(
            painter=painter,
            fm=fm_current,
            rect=rect,
            t=t,
        )

        try:
            reveal_duration = float(self.config.get("particle_reveal_duration", 1.0))
        except Exception:
            reveal_duration = 1.0
        reveal_duration = max(0.05, reveal_duration)

        phrase_start = self._phrase_start_t if self._phrase_start_t is not None else t
        phrase_elapsed = max(0.0, t - phrase_start)
        reveal_progress = max(0.0, min(1.0, phrase_elapsed / reveal_duration))
        reveal_eased = 1.0 - pow(1.0 - reveal_progress, 3.0)

        # Silence fade (fixed duration, no parameter exposed)
        silence_mult = 1.0
        if self._silence_active and self._silence_start_t is not None:
            dt_silence = max(0.0, t - self._silence_start_t)
            silence_fade = 0.7  # fixed fade duration in seconds
            frac = max(0.0, min(1.0, dt_silence / silence_fade))
            silence_mult = 1.0 - frac

        # Style mode (classic / neon)
        style_mode = str(self.config.get("style_mode", "classic") or "classic")
        is_neon = (style_mode == "neon")

        # -------------------------------------------------------------- #
        # Draw particles                                                 #
        # -------------------------------------------------------------- #
        painter.save()

        try:
            trail_opacity = float(self.config.get("particle_trail_opacity", 0.9))
        except Exception:
            trail_opacity = 0.9
        trail_opacity = max(0.1, min(1.0, trail_opacity))

        # Allow live text positioning without regenerating particles:
        # shift everything by the delta from the cached anchor.
        cur_ax, cur_ay = self._get_text_anchor_xy(rect)
        offset_x = float(cur_ax - self._layout_anchor_x)
        offset_y = float(cur_ay - self._layout_anchor_y)

        backend = str(self.config.get("render_backend", "points_fast") or "points_fast")
        antialias = bool(self.config.get("antialiasing", False))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, bool(antialias))

        if backend == "ellipses":
            self._draw_particles_ellipses(
                painter=painter,
                t=t,
                reveal_eased=reveal_eased,
                silence_mult=silence_mult,
                is_neon=is_neon,
                trail_opacity=trail_opacity,
                offset_x=offset_x,
                offset_y=offset_y,
            )
        else:
            self._draw_particles_points(
                painter=painter,
                t=t,
                reveal_eased=reveal_eased,
                silence_mult=silence_mult,
                is_neon=is_neon,
                trail_opacity=trail_opacity,
                offset_x=offset_x,
                offset_y=offset_y,
                pro=(backend == "points_pro"),
            )
        painter.restore()
        painter.end()
