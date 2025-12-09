# olaf_app/cover_visualization_api.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .visualization_api import PluginParameter


@dataclass
class FrameFeatures:
    """
    Per-frame audio features for a cover effect.

    The host is free to add more keys, but at minimum we expect:
      - amp: float in [0, 1], main amplitude driving this effect.
    """
    amp: float = 0.0
    # Optional arbitrary data; always a dict so plugins can safely call .get()
    extra: Dict[str, Any] = field(default_factory=dict)



class BaseCoverEffect(ABC):
    """
    Base class for 2D cover-based effects.

    A cover effect:
      - Receives an RGB frame (uint8, HxWx3).
      - Receives time `t` in seconds (for time-based modulation).
      - Receives audio features (FrameFeatures) for the stem(s) routed to it.
      - Returns a new RGB frame with the effect applied.

    This is meant for OFFLINE rendering (export pipeline), not for the 3D
    real-time preview widget system.
    """

    # Metadata (to be overridden by subclasses)
    effect_id: str = "base_cover_effect"
    effect_name: str = "Base cover effect"
    effect_description: str = ""
    effect_author: str = "Unknown"
    effect_version: str = "0.1.0"

    # How many audio inputs (stems) this effect can conceptually use
    # (used by the future routing UI; the host is free to ignore it).
    effect_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # User configuration (from UI / project JSON)
        self.config: Dict[str, Any] = config or {}

        # Deterministic RNG for glitchy effects, seeded from config if provided
        seed = int(self.config.get("random_seed", 2024))
        self._rng = np.random.default_rng(seed=seed)

    # ------------------------------------------------------------------ #
    # Parameter specification                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Return a schema describing configurable parameters for this effect.

        Keys are parameter names, values are PluginParameter instances.
        """
        return {}

    # ------------------------------------------------------------------ #
    # Lifecycle hooks                                                    #
    # ------------------------------------------------------------------ #

    def on_sequence_start(self, duration: float, fps: int) -> None:
        """
        Optional hook called at the start of a rendered sequence.

        The host can call this once before rendering the first frame to
        let the effect prepare internal state, precompute masks, etc.
        """
        # Default implementation does nothing.
        return

    # ------------------------------------------------------------------ #
    # Core processing                                                    #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def apply_to_frame(
        self,
        frame_rgb: np.ndarray,
        t: float,
        features: FrameFeatures,
    ) -> np.ndarray:
        """
        Apply the effect to a single RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray
            Input image as uint8 array (H, W, 3) in RGB order.
        t : float
            Current time in seconds since the start of the sequence.
        features : FrameFeatures
            Audio features (e.g. RMS/energy) for the stem(s) driving this effect.

        Returns
        -------
        np.ndarray
            New RGB frame with the effect applied. The default contract
            expects uint8 (H, W, 3) in RGB order.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # State save/load (for project persistence)                          #
    # ------------------------------------------------------------------ #

    def save_state(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot of the configuration.

        The host is free to extend this with extra internal state if needed.
        """
        return dict(self.config)

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore configuration from a previously saved state.
        """
        self.config.update(state)
