from __future__ import annotations

"""
Discovery and instantiation of lyrics visualization plugins.

Plugins live in the `lyrics_visuals/` folder and must define at least
one subclass of BaseLyricsVisualization.
"""

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from .lyrics_visualization_api import BaseLyricsVisualization
from .visualization_api import PluginParameter


LYRICS_VISUALS_ROOT = Path("lyrics_visuals")


@dataclass
class LyricsVisualizationInfo:
    """
    Metadata for a single lyrics visualization plugin.
    """

    plugin_id: str
    name: str
    description: str
    author: str
    version: str
    parameters: Dict[str, PluginParameter]

    cls: Type[BaseLyricsVisualization]
    module_path: Path
    module_name: str


class LyricsVisualizationsManager:
    """
    Discover and instantiate lyrics visualization plugins.

    This manager is intentionally similar to CoverVisualizationsManager
    so the rest of the app can follow the same patterns.
    """

    def __init__(self, visuals_root: Optional[Path] = None) -> None:
        self.visuals_root: Path = visuals_root or LYRICS_VISUALS_ROOT
        self._visuals: Dict[str, LyricsVisualizationInfo] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """
        Rescan the lyrics_visuals/ folder and rebuild the plugins registry.

        Broken plugins are ignored instead of crashing the application.
        """
        self._visuals.clear()

        root = self.visuals_root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Silently ignore folder creation errors; discovery will just be empty.
            return

        for path in sorted(root.glob("*.py")):
            self._load_module(path)

    def _load_module(self, module_path: Path) -> None:
        """
        Import a single Python file and register all BaseLyricsVisualization
        subclasses defined inside.
        """
        module_name = f"olaf_app.lyrics_visuals.{module_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except Exception:
            # Do not let a broken plugin crash the whole app
            return

        from .lyrics_visualization_api import BaseLyricsVisualization as _Base

        for _, obj in inspect.getmembers(module, inspect.isclass):
            # Only consider concrete subclasses of BaseLyricsVisualization
            if not issubclass(obj, _Base):
                continue
            if obj is _Base:
                continue

            plugin_id = getattr(obj, "plugin_id", None)
            if not plugin_id:
                continue

            # Do not allow duplicates silently; last one wins.
            try:
                parameters = obj.parameters()
            except Exception:
                parameters = {}

            info = LyricsVisualizationInfo(
                plugin_id=plugin_id,
                name=getattr(obj, "plugin_name", plugin_id),
                description=getattr(obj, "plugin_description", ""),
                author=getattr(obj, "plugin_author", "Unknown"),
                version=getattr(obj, "plugin_version", "0.1.0"),
                parameters=parameters,
                cls=obj,
                module_path=module_path,
                module_name=module_name,
            )
            self._visuals[plugin_id] = info

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def available_visuals(self) -> List[LyricsVisualizationInfo]:
        """
        Return the list of discovered plugins, sorted by human-readable name.
        """
        return sorted(self._visuals.values(), key=lambda info: info.name.lower())

    def get_info(self, plugin_id: str) -> Optional[LyricsVisualizationInfo]:
        """Return metadata for a plugin, or None if not found."""
        return self._visuals.get(plugin_id)

    # ------------------------------------------------------------------
    # Instantiation
    # ------------------------------------------------------------------
    def create_instance(
        self,
        plugin_id: str,
        config: Optional[Dict[str, Any]] = None,
        parent: Optional[object] = None,
    ) -> Optional[BaseLyricsVisualization]:
        """
        Instantiate the plugin with the given id, using the provided configuration.
        """
        info = self._visuals.get(plugin_id)
        if info is None:
            return None
        try:
            instance = info.cls(config=config or {}, parent=parent)  # type: ignore[arg-type]
        except Exception:
            return None
        return instance
