# visualizations_manager.py
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from .visualization_api import BaseVisualization, PluginParameter

# Root folder where visualization plugins live (Python files)
# e.g. visuals/neons_ribbon.py, visuals/circular_double.py, etc.
VISUALS_ROOT = Path("visuals")


@dataclass
class VisualizationPluginInfo:
    """
    Metadata describing a single visualization plugin.

    This is what the UI (VisualizationsTab) consumes:
      - plugin_id: stable identifier used in project.json
      - name: human-readable name
      - description: short explanation for the user
      - author: plugin author (for display only)
      - version: semantic version string
      - max_inputs: how many stems this plugin can accept
      - parameters: configuration schema (name -> PluginParameter)

    The module_path / class_name fields are internal, used to re-instantiate
    the plugin class when needed.
    """

    plugin_id: str
    name: str
    description: str
    author: str
    version: str
    max_inputs: int
    parameters: Dict[str, PluginParameter]

    module_path: Path
    class_name: str
    module_name: str  # internal Python module name used by importlib


class VisualizationManager:
    """
    Discovers visualization plugins on disk and exposes them to the UI.

    It looks for .py files in VISUALS_ROOT and registers all classes that
    inherit from BaseVisualization.
    """

    def __init__(self, visuals_root: Optional[Path] = None) -> None:
        self.visuals_root: Path = visuals_root or VISUALS_ROOT
        self._plugins: Dict[str, VisualizationPluginInfo] = {}

    @property
    def plugins(self) -> Dict[str, VisualizationPluginInfo]:
        return self._plugins

    # ------------------------------------------------------------------ #
    # Discovery                                                          #
    # ------------------------------------------------------------------ #

    def discover_plugins(self) -> None:
        """
        Rescan the visuals directory and rebuild the internal plugin list.
        Broken plugins are skipped silently so that one bad file does not
        break the entire application.
        """
        self._plugins.clear()

        root = self.visuals_root
        if not root.exists() or not root.is_dir():
            return

        for path in sorted(root.glob("*.py")):
            # Skip private / special files
            if path.name.startswith("_") or path.name == "__init__.py":
                continue
            self._load_plugin_file(path)

    def _load_plugin_file(self, path: Path) -> None:
        """
        Load a single .py file and register any BaseVisualization subclasses.
        Broken plugins are skipped silently.
        """
        # We give each plugin file a unique module name under a private namespace
        module_name = f"olaf_visuals.{path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception:
            # Do not let a single broken plugin crash the app
            return

        # Find all subclasses of BaseVisualization defined in this module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseVisualization):
                continue
            if obj is BaseVisualization:
                continue

            info = self._build_info_from_class(
                cls=obj,
                module_path=path,
                module_name=module_name,
            )
            if info is None:
                continue

            # First plugin wins if there is a duplicate plugin_id
            if info.plugin_id in self._plugins:
                continue

            self._plugins[info.plugin_id] = info

    def _build_info_from_class(
        self,
        cls: Type[BaseVisualization],
        module_path: Path,
        module_name: str,
    ) -> Optional[VisualizationPluginInfo]:
        """
        Extract metadata from a plugin class.

        Expected class attributes:
          - plugin_id (str)
          - plugin_name (str)
          - plugin_description (str)
          - plugin_author (str)
          - plugin_version (str)
          - plugin_max_inputs (int)
          - @classmethod parameters() -> Dict[str, PluginParameter]
        """
        # plugin_id is mandatory: if missing, we skip the plugin
        plugin_id = getattr(cls, "plugin_id", None)
        if not plugin_id or not isinstance(plugin_id, str):
            return None

        name = getattr(cls, "plugin_name", plugin_id)
        description = getattr(cls, "plugin_description", "") or ""
        author = getattr(cls, "plugin_author", "Unknown")
        version = getattr(cls, "plugin_version", "1.0")
        max_inputs = int(getattr(cls, "plugin_max_inputs", 1))

        parameters: Dict[str, PluginParameter] = {}
        try:
            if hasattr(cls, "parameters") and callable(getattr(cls, "parameters")):
                parameters = cls.parameters()  # type: ignore[assignment]
        except Exception:
            parameters = {}

        return VisualizationPluginInfo(
            plugin_id=plugin_id,
            name=name,
            description=description,
            author=author,
            version=version,
            max_inputs=max_inputs,
            parameters=parameters,
            module_path=module_path,
            class_name=cls.__name__,
            module_name=module_name,
        )

    # ------------------------------------------------------------------ #
    # Public API used by VisualizationsTab                               #
    # ------------------------------------------------------------------ #

    def list_plugins(self) -> List[VisualizationPluginInfo]:
        """
        Return all discovered plugins as a list, sorted by display name.
        """
        return sorted(self._plugins.values(), key=lambda p: p.name.lower())

    def get_plugin(self, plugin_id: str) -> Optional[VisualizationPluginInfo]:
        """
        Return metadata for a single plugin, or None if unknown.
        """
        return self._plugins.get(plugin_id)

    def create_instance(
        self,
        plugin_id: str,
        config: Optional[Dict[str, object]] = None,
    ) -> Optional[BaseVisualization]:
        """
        Instantiate a plugin by its id.

        The returned object is a BaseVisualization subclass with the given
        configuration dict attached to its `config` attribute.
        """
        info = self.get_plugin(plugin_id)
        if info is None:
            return None

        # Try to reuse the module if already loaded, otherwise import again
        module = sys.modules.get(info.module_name)
        if module is None:
            try:
                spec = importlib.util.spec_from_file_location(
                    info.module_name, info.module_path
                )
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[info.module_name] = module
                spec.loader.exec_module(module)
            except Exception:
                return None

        cls = getattr(module, info.class_name, None)
        if cls is None:
            return None

        try:
            instance: BaseVisualization = cls(config=config or {})
        except Exception:
            return None

        return instance
