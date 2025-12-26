from __future__ import annotations

"""
Shared text-style helpers for all lyrics visualization plugins.

This module centralizes:
  * default config values for text rendering (font, color, outline, shadow,
    background box),
  * the PluginParameter definitions used by the UI to build controls,
  * small runtime helpers to build QFont / QColor objects and draw styled text.

All comments are in English as this project is intended to be shared on GitHub.
"""

from typing import Any, Dict, List
from pathlib import Path

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QBrush, QPen, QPainterPath
from olaf_app.visualization_api import PluginParameter

# ---------------------------------------------------------------------------
# Local font loading for lyrics (defensive against import order issues)
# ---------------------------------------------------------------------------

_FONTS_LOADED_FOR_LYRICS = False


def _ensure_custom_fonts_loaded_for_lyrics() -> None:
    """
    Make sure that custom fonts from the project-level `fonts/` directory
    and the package-local `olaf_app/fonts/` directory are registered in
    QFontDatabase.

    This is intentionally redundant with gui.load_custom_fonts() so that
    lyrics plugins still see the fonts even if this module is used in a
    different context (tests, standalone tools, etc.).
    """
    global _FONTS_LOADED_FOR_LYRICS
    if _FONTS_LOADED_FOR_LYRICS:
        return

    _FONTS_LOADED_FOR_LYRICS = True

    try:
        here = Path(__file__).resolve()
    except Exception:
        return

    # olaf_app package root = .../olaf_app
    package_root = here.parent

    candidate_dirs = [
        package_root.parent / "fonts", # project_root/fonts 
    ]

    for fonts_dir in candidate_dirs:
        if not fonts_dir.is_dir():
            continue

        for pattern in ("*.ttf", "*.otf"):
            for font_path in fonts_dir.glob(pattern):
                try:
                    QFontDatabase.addApplicationFont(str(font_path))
                except Exception:
                    # Never crash UI because a font file is broken
                    pass
# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def apply_default_text_style_config(config: Dict[str, Any]) -> None:
    """
    Ensure that the given config dict contains all expected text-style keys.

    This is deliberately defensive so that older project files or partially
    filled configs still work without crashing.
    """
    # Basic font
    config.setdefault("font_family", "")

    # Font sizing strategy:
    # - New projects store a relative size (fraction of render height) in
    #   `font_size_rel` (Option A).
    # - Older projects may only contain a fixed `font_size` (legacy).
    #
    # We keep `font_size` for backward compatibility, but the UI is driven by
    # `font_size_rel` and the host can convert it to a concrete pixel size at
    # render time.
    legacy_px = config.get("font_size", None)
    if "font_size_rel" not in config and isinstance(legacy_px, (int, float)):
        try:
            config["font_size_rel"] = float(legacy_px) / 1080.0
        except Exception:
            pass

    config.setdefault("font_size_rel", 0.06)  # ~6% of frame height (1080p -> ~65px)
    config.setdefault("font_size", 40)        # legacy / fallback
    config.setdefault("font_bold", False)
    config.setdefault("font_color", "#ffffff")
    config.setdefault("capitalize_all", False)
# Outline
    config.setdefault("text_outline_enabled", True)
    config.setdefault("text_outline_color", "#000000")
    config.setdefault("text_outline_width", 2.0)

    # Shadow
    config.setdefault("text_shadow_enabled", True)
    config.setdefault("text_shadow_color", "#000000")
    config.setdefault("text_shadow_offset_x", 3)
    config.setdefault("text_shadow_offset_y", 3)

    # Background box
    config.setdefault("text_box_enabled", False)
    config.setdefault("text_box_color", "#000000")
    config.setdefault("text_box_alpha", 0.55)
    config.setdefault("text_box_padding", 0.35)


# ---------------------------------------------------------------------------
# Shared PluginParameter definitions
# ---------------------------------------------------------------------------


def text_style_parameters() -> Dict[str, PluginParameter]:
    """
    Return a dictionary of shared PluginParameter definitions.

    The list of font families is built from QFontDatabase, which includes:
      * system fonts
      * any application fonts loaded via QFontDatabase.addApplicationFont()
        (for example the files in project_root/fonts and olaf_app/fonts).

    Families are sorted alphabetically so that the combo box is stable and
    easy to browse.
    """
    # Make sure our custom fonts are registered, even if gui.load_custom_fonts()
    # has not been called yet or was called in a different code path.
    _ensure_custom_fonts_loaded_for_lyrics()

    families: List[str]

    try:
        db = QFontDatabase()
        all_families = {str(f) for f in db.families()}
        if all_families:
            families = sorted(all_families)
        else:
            raise RuntimeError("QFontDatabase returned no families")
    except Exception:
        # Very defensive fallback; this should rarely be used.
        fallback = [
            "Arial",
            "Times New Roman",
            "Courier New",
            "Verdana",
            "Tahoma",
            "Segoe UI",
            "Roboto",
            "Noto Sans",
            "DejaVu Sans",
        ]
        families = sorted(set(fallback))

    default_family = families[0] if families else "Arial"

    return {
        # Basic font
        "font_family": PluginParameter(
            name="font_family",
            label="Font family",
            type="enum",
            default=default_family,
            choices=families,
            description=(
                "Font family used for the lyrics text. Includes system fonts "
                "and any application fonts loaded at startup."
            ),
        ),
        "font_size_rel": PluginParameter(
            name="font_size_rel",
            label="Font size (relative)",
            type="float",
            default=0.06,
            minimum=0.01,
            maximum=0.20,
            step=0.005,
            description=(
                "Font size expressed as a fraction of the render height "
                "(e.g. 0.06 ≈ 6% of the frame height). This makes text scale "
                "automatically with the output resolution."
            ),
        ),
        "font_bold": PluginParameter(
            name="font_bold",
            label="Bold text",
            type="bool",
            default=False,
            description="If checked, the lyrics text is rendered in bold.",
        ),
        "font_color": PluginParameter(
            name="font_color",
            label="Text color",
            type="color",
            default="#ffffff",
            description="Color used for the lyrics text.",
        ),
        "capitalize_all": PluginParameter(
            name="capitalize_all",
            label="Capitalize everything",
            type="bool",
            default=False,
            description="If checked, all lyrics are rendered in ALL CAPS.",
        ),

        # Outline
        "text_outline_enabled": PluginParameter(
            name="text_outline_enabled",
            label="Outline enabled",
            type="bool",
            default=True,
            description="Draw an outline around the text.",
        ),
        "text_outline_color": PluginParameter(
            name="text_outline_color",
            label="Outline color",
            type="color",
            default="#000000",
            description="Color of the text outline.",
        ),
        "text_outline_width": PluginParameter(
            name="text_outline_width",
            label="Outline width (px)",
            type="float",
            default=2.0,
            minimum=0.5,
            maximum=10.0,
            step=0.5,
            description="Thickness of the outline stroke around the text.",
        ),

        # Shadow
        "text_shadow_enabled": PluginParameter(
            name="text_shadow_enabled",
            label="Shadow enabled",
            type="bool",
            default=True,
            description="Draw a drop shadow behind the text.",
        ),
        "text_shadow_color": PluginParameter(
            name="text_shadow_color",
            label="Shadow color",
            type="color",
            default="#000000",
            description="Color of the text shadow.",
        ),
        "text_shadow_offset_x": PluginParameter(
            name="text_shadow_offset_x",
            label="Shadow offset X (px)",
            type="int",
            default=3,
            minimum=-50,
            maximum=50,
            step=1,
            description="Horizontal offset of the text shadow in pixels.",
        ),
        "text_shadow_offset_y": PluginParameter(
            name="text_shadow_offset_y",
            label="Shadow offset Y (px)",
            type="int",
            default=3,
            minimum=-50,
            maximum=50,
            step=1,
            description="Vertical offset of the text shadow in pixels.",
        ),

        # Background box
        "text_box_enabled": PluginParameter(
            name="text_box_enabled",
            label="Background box enabled",
            type="bool",
            default=False,
            description="If checked, draws a box behind the text.",
        ),
        "text_box_color": PluginParameter(
            name="text_box_color",
            label="Background box color",
            type="color",
            default="#000000",
            description="Color of the box behind the text.",
        ),
        "text_box_alpha": PluginParameter(
            name="text_box_alpha",
            label="Background box opacity",
            type="float",
            default=0.55,
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            description="Opacity of the background box (0 = transparent, 1 = opaque).",  # noqa: E501
        ),
        "text_box_padding": PluginParameter(
            name="text_box_padding",
            label="Background box padding (rel.)",
            type="float",
            default=0.35,
            minimum=0.0,
            maximum=2.0,
            step=0.05,
            description="Padding around the text, relative to font height.",
        ),
    }


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def build_qfont_from_config(
    config: Dict[str, Any],
    painter: QPainter,
    point_size: int,
) -> QFont:
    """
    Build a QFont from the style options stored in *config*.

    The *painter* is only used to retrieve a sensible default font when
    no family is specified in the config.

    Important:
        The `point_size` argument is interpreted as a **pixel size**, not a
        typographic point size.

        Using point sizes makes the final pixel height depend on the device DPI
        (e.g. screen widgets vs offscreen QImage). Since Olaf stores / computes
        font sizing in pixels (via `font_size_rel * render_height`), we must
        use `QFont.setPixelSize(...)` for consistent results between preview and
        export.
    """
    # Defensive clamp: Qt behaves poorly with <= 0 sizes.
    px = int(point_size)
    if px <= 0:
        px = 1

    family = str(config.get("font_family") or "").strip()
    if family:
        font = QFont(family)
        font.setPixelSize(px)
    else:
        # Start from the painter's current font to inherit platform defaults,
        # but enforce an explicit pixel size.
        font = QFont(painter.font())
        font.setPixelSize(px)

    if bool(config.get("font_bold", False)):
        font.setBold(True)
    else:
        font.setBold(False)

    return font


def font_color_from_config(config: Dict[str, Any]) -> QColor:
    """
    Return the QColor to use for the main text.
    """
    value = str(config.get("font_color") or "").strip()
    if not value:
        return QColor("#ffffff")
    color = QColor(value)
    if not color.isValid():
        return QColor("#ffffff")
    return color


def _apply_outline_and_shadow(
    painter: QPainter,
    x: float,
    y: float,
    text: str,
    config: Dict[str, Any],
    base_color: QColor,
    global_opacity: float = 1.0,
) -> None:
    """
    Internal helper: draw shadow + outline + fill for the given text.

    We apply the caller's opacity *to the colors* and draw with painter opacity
    forced to 1.0. This guarantees that per-word fading impacts fill, outline
    and shadow consistently, and avoids outline overdraw becoming visually
    too opaque compared to the fill.
    """
    try:
        global_opacity_f = float(global_opacity)
    except Exception:
        global_opacity_f = 1.0
    global_opacity_f = max(0.0, min(1.0, global_opacity_f))

    def _scale_alpha(c: QColor) -> QColor:
        cc = QColor(c)
        cc.setAlphaF(max(0.0, min(1.0, cc.alphaF() * global_opacity_f)))
        return cc

    # Build a vector path for the glyphs: one stroke, one fill (better alpha behavior).
    path = QPainterPath()
    path.addText(float(x), float(y), painter.font(), text)

    # Shadow (filled path).
    if bool(config.get("text_shadow_enabled", False)):
        shadow_color = QColor(str(config.get("text_shadow_color") or "#000000"))
        shadow_color = _scale_alpha(shadow_color)
        try:
            shadow_dx = float(config.get("text_shadow_dx", 2.0))
            shadow_dy = float(config.get("text_shadow_dy", 2.0))
        except Exception:
            shadow_dx, shadow_dy = 2.0, 2.0

        if shadow_color.alpha() > 0:
            painter.save()
            painter.translate(shadow_dx, shadow_dy)
            painter.fillPath(path, QBrush(shadow_color))
            painter.restore()

    # Outline (single stroked path).
    if bool(config.get("text_outline_enabled", False)):
        outline_color = QColor(str(config.get("text_outline_color") or "#000000"))
        outline_color = _scale_alpha(outline_color)
        try:
            width = float(config.get("text_outline_width", 2.0))
        except Exception:
            width = 2.0

        if width > 0.01 and outline_color.alpha() > 0:
            pen = QPen(outline_color)
            pen.setWidthF(width)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.strokePath(path, pen)

    # Main fill.
    painter.fillPath(path, QBrush(_scale_alpha(base_color)))



def draw_styled_text(
    painter: QPainter,
    x: float,
    y: float,
    text: str,
    config: Dict[str, Any],
    base_font: QFont,
    base_color: QColor,
) -> None:
    """
    Draw *text* at (x, y) with outline / shadow / background box.

    This function centralizes the styling logic so all lyrics plugins get a
    consistent appearance.
    """
    # Capitalization (optional)
    if bool(config.get("capitalize_all", False)):
        text_to_draw = text.upper()
    else:
        text_to_draw = text

    painter.save()
    painter.setFont(base_font)

    # The caller may set painter opacity for per-word fading.
    # We capture it once and apply it to colors (fill/outline/shadow/box) to
    # keep all components fading consistently.
    try:
        global_opacity = float(painter.opacity())
    except Exception:
        global_opacity = 1.0
    global_opacity = max(0.0, min(1.0, global_opacity))
    if global_opacity < 0.999:
        painter.setOpacity(1.0)


    fm = QFontMetrics(base_font)
    text_width = fm.horizontalAdvance(text_to_draw)
    text_height = fm.height()

    # Background box
    if bool(config.get("text_box_enabled", False)):
        box_color = QColor(str(config.get("text_box_color") or "#000000"))
        if not box_color.isValid():
            box_color = QColor("#000000")
        alpha = float(config.get("text_box_alpha", 0.55))
        alpha_clamped = max(0.0, min(1.0, alpha))
        box_color.setAlphaF(alpha_clamped * global_opacity)

        padding_rel = float(config.get("text_box_padding", 0.35))
        pad = max(0.0, padding_rel) * text_height

        box_rect = QRectF(
            x - pad,
            y - text_height - pad * 0.5,
            text_width + 2 * pad,
            text_height + pad,
        )
        painter.fillRect(box_rect, box_color)

    # Outline + shadow + fill
    _apply_outline_and_shadow(
        painter=painter,
        x=x,
        y=y,
        text=text_to_draw,
        config=config,
        global_opacity=global_opacity,
        base_color=base_color,
    )

    painter.restore()
