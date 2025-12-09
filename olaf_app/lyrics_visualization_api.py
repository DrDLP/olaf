from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from PyQt6.QtWidgets import QWidget
from .visualization_api import PluginParameter

# Cache for custom fonts loaded from the local "fonts" folder
_CUSTOM_FONTS_LOADED = False
_CUSTOM_FONT_FAMILIES: list[str] = []

@dataclass
class LyricsFrameContext:
    """
    Per-frame state for a lyrics visualization.

    The host (Olaf) is responsible for:
      * computing the absolute playback time `t` (seconds),
      * deriving the current phrase and word from the alignment (phrases/words JSON),
      * computing audio features such as `amp` from an RMS envelope,
      * building this dataclass and passing it to the plugin.
    """

    # Global playback time (seconds since the beginning of the song).
    t: float

    # Normalized audio amplitude in [0.0, 1.0] for the configured audio source.
    amp: float

    # Phrase-level information
    phrase_index: Optional[int]
    local_phrase_time: float  # seconds since the beginning of the phrase
    phrase_duration: float    # total phrase duration in seconds (>= 0.0)
    text_full_line: str

    # Word-level information (global index in the words list)
    word_index: Optional[int]
    text_active_word: Optional[str]

    # Character offsets of the active word within text_full_line.
    # These are None if no active word is present or if the mapping
    # could not be resolved unambiguously.
    word_char_start: Optional[int]
    word_char_end: Optional[int]


class BaseLyricsVisualization(QWidget):
    """
    Base class for all lyrics visualization plugins.

    A plugin is a QWidget that draws itself (usually in paintEvent)
    and reacts to the stream of LyricsFrameContext values passed by
    the host through update_frame().
    """

    plugin_id: str = "base_lyrics_visualization"
    plugin_name: str = "Base lyrics visualization"
    plugin_description: str = ""
    plugin_author: str = "Unknown"
    plugin_version: str = "0.1.0"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        # Arbitrary user configuration declared via parameters().
        self.config: Dict[str, Any] = config or {}
        # Last frame context, stored so paintEvent() can access it.
        self._last_ctx: Optional[LyricsFrameContext] = None

    # ------------------------------------------------------------------
    # Parameters metadata
    # ------------------------------------------------------------------
    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Return the parameter specification for this plugin.

        Subclasses should override and declare every configurable
        parameter as a PluginParameter. The host will then build
        the corresponding Qt widgets automatically.
        """
        return {}

    # ------------------------------------------------------------------
    # State (de)serialization
    # ------------------------------------------------------------------
    def save_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the plugin configuration.

        The host will store this in the project JSON under
        lyrics_visual_parameters.
        """
        return dict(self.config)

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore configuration from a previously saved state."""
        if not isinstance(state, dict):
            return
        self.config.update(state)

    # ------------------------------------------------------------------
    # Frame update API
    # ------------------------------------------------------------------
    def update_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Called by the host whenever the playback position or audio
        features change.

        Default implementation:
          * store the context,
          * call on_frame() so subclasses can update internal state,
          * schedule a repaint (paintEvent will read self._last_ctx).
        """
        self._last_ctx = ctx
        self.on_frame(ctx)
        self.update()

    def current_context(self) -> Optional[LyricsFrameContext]:
        """Return the last context passed to update_frame(), if any."""
        return self._last_ctx

    @abstractmethod
    def on_frame(self, ctx: LyricsFrameContext) -> None:
        """
        Hook called after every update_frame().

        Subclasses typically:
          * read ctx.t, ctx.amp, ctx.text_full_line, ctx.text_active_word,
          * update internal animation state (e.g. fades, offsets),
          * call self.update() (already done in BaseLyricsVisualization).
        """
        raise NotImplementedError("on_frame() must be implemented by subclasses")