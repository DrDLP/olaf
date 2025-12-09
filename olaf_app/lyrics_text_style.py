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

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter
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
    config.setdefault("font_size", 40)
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
        "font_size": PluginParameter(
            name="font_size",
            label="Base font size",
            type="int",
            default=40,
            minimum=16,
            maximum=200,
            step=1,
            description="Base point size used to draw the lyrics text.",
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
    """
    family = str(config.get("font_family") or "").strip()
    if family:
        font = QFont(family, point_size)
    else:
        font = painter.font()
        font.setPointSize(point_size)

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
) -> None:
    """
    Internal helper: draw shadow + outline for the given text.

    This expects the painter to already be configured with the desired font.
    """
    # Shadow
    if bool(config.get("text_shadow_enabled", False)):
        shadow_color = QColor(str(config.get("text_shadow_color") or "#000000"))
        if not shadow_color.isValid():
            shadow_color = QColor("#000000")
        dx = int(config.get("text_shadow_offset_x", 3))
        dy = int(config.get("text_shadow_offset_y", 3))
        painter.setPen(shadow_color)
        painter.drawText(int(x + dx), int(y + dy), text)

    # Outline
    if bool(config.get("text_outline_enabled", False)):
        outline_color = QColor(str(config.get("text_outline_color") or "#000000"))
        if not outline_color.isValid():
            outline_color = QColor("#000000")
        width = float(config.get("text_outline_width", 2.0))
        stroke_steps = max(1, int(width))

        painter.setPen(outline_color)
        for dx in range(-stroke_steps, stroke_steps + 1):
            for dy in range(-stroke_steps, stroke_steps + 1):
                if dx == 0 and dy == 0:
                    continue
                painter.drawText(int(x + dx), int(y + dy), text)

    # Main text fill
    painter.setPen(base_color)
    painter.drawText(int(x), int(y), text)


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
        box_color.setAlphaF(alpha_clamped)

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
        base_color=base_color,
    )

    painter.restore()
