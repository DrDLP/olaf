from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import QWidget


@dataclass
class PluginParameter:
    """
    Description of a single configurable parameter for a visualization plugin.

    For numeric parameters (type == "int" or "float"), the optional `step`
    value controls the increment used by slider widgets in the UI.
    """
    name: str
    label: str
    type: str  # "int", "float", "bool", "enum"
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[List[Any]] = None
    step: Optional[float] = None
    description: str = ""


class BaseVisualization(ABC):
    """
    Base class that every visualization plugin must inherit from.
    The host (Olaf) will discover subclasses of this class.
    """

    plugin_id: str = "base_visualization"
    plugin_name: str = "Base visualization"
    plugin_description: str = ""
    plugin_author: str = "Unknown"
    plugin_version: str = "0.1.0"
    # How many independent audio inputs (stems) this plugin supports. Hard-limited to 5.
    plugin_max_inputs: int = 1

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    @classmethod
    def parameters(cls) -> Dict[str, PluginParameter]:
        """
        Return a dictionary of parameter specifications for this plugin.

        Keys must be parameter names, values PluginParameter instances.
        """
        return {}

    @abstractmethod
    def create_widget(self, parent: Optional[QWidget] = None) -> QWidget:
        """
        Return the QWidget that will render the visualization.
        """
        raise NotImplementedError

    @abstractmethod
    def on_audio_features(self, features: Dict[str, Any]) -> None:
        """
        Called regularly by the host with audio features for each input.
        """
        raise NotImplementedError

    def on_activate(self) -> None:
        """Called when the visualization is activated in the UI."""
        pass

    def on_deactivate(self) -> None:
        """Called when the visualization is deactivated or removed."""
        pass

    def on_deactivate(self) -> None:
        """Called when the visualization is deactivated or removed."""
        pass

    def apply_preview_settings(self, width: int, height: int, fps: int) -> None:
        """
        Optional hook called by the host to tell the plugin what preview
        size and nominal frame rate should be used.

        Implementations can ignore this or use it to adapt internal timers
        and rendering resolution. The default implementation does nothing.
        """
        return

    def save_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the plugin configuration."""
        return dict(self.config)

    def save_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the plugin configuration."""
        return dict(self.config)

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore configuration from a previously saved state."""
        self.config.update(state)
