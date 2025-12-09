# cover_visualizations_manager.py
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from .cover_visualization_api import BaseCoverEffect
from .visualization_api import PluginParameter

# Root folder where 2D cover effects live (Python files)
# e.g. cover_visuals/glitch_slices.py
COVER_VISUALS_ROOT = Path("cover_visuals")


@dataclass
class CoverEffectInfo:
    """
    Metadata describing a single cover-based 2D effect.

    This is what the 2D visualizations UI (CoverVisualizationsTab) will consume:
      - effect_id: stable identifier used in project.json
      - name: human-readable name
      - description: short explanation for the user
      - author: effect author (for display only)
      - version: semantic version string
      - max_inputs: how many stems this effect can conceptually use
      - parameters: configuration schema (name -> PluginParameter)

    The module_path / class_name / module_name fields are internal, used to
    re-instantiate the effect class when needed.
    """

    effect_id: str
    name: str
    description: str
    author: str
    version: str
    max_inputs: int
    parameters: Dict[str, PluginParameter]

    module_path: Path
    class_name: str
    module_name: str  # internal Python module name used by importlib


class CoverVisualizationsManager:
    """
    Discovers cover-based 2D effects on disk and exposes them to the UI.

    It looks for .py files in COVER_VISUALS_ROOT and registers all classes that
    inherit from BaseCoverEffect.
    """

    def __init__(self, visuals_root: Optional[Path] = None) -> None:
        # 'visuals_root' name is kept for symmetry with the 3D manager.
        self.visuals_root: Path = visuals_root or COVER_VISUALS_ROOT
        self._effects: Dict[str, CoverEffectInfo] = {}

    @property
    def effects(self) -> Dict[str, CoverEffectInfo]:
        """Return the internal mapping effect_id -> CoverEffectInfo."""
        return self._effects

    # ------------------------------------------------------------------ #
    # Discovery                                                          #
    # ------------------------------------------------------------------ #

    def discover_plugins(self) -> None:
        """
        Rescan the cover visuals directory and rebuild the internal effect list.

        Broken effect files are skipped silently so that one bad file does not
        break the entire application.
        """
        self._effects.clear()

        root = self.visuals_root
        if not root.exists() or not root.is_dir():
            return

        for path in sorted(root.glob("*.py")):
            # Skip private / special files
            if path.name.startswith("_") or path.name == "__init__.py":
                continue
            self._load_effect_file(path)

    def _load_effect_file(self, path: Path) -> None:
        """
        Load a single .py file and register any BaseCoverEffect subclasses.

        Broken or incompatible effect files are skipped silently.
        """
        # We give each effect file a unique module name under a private namespace
        module_name = f"olaf_cover_visuals.{path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except Exception:
            # Do not let a single broken plugin crash the app
            return

        # Find all subclasses of BaseCoverEffect defined in this module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseCoverEffect):
                continue
            if obj is BaseCoverEffect:
                continue

            info = self._build_info_from_class(
                cls=obj,
                module_path=path,
                module_name=module_name,
            )
            if info is None:
                continue

            # First effect wins if there is a duplicate effect_id
            if info.effect_id in self._effects:
                continue

            self._effects[info.effect_id] = info

    def _build_info_from_class(
        self,
        cls: Type[BaseCoverEffect],
        module_path: Path,
        module_name: str,
    ) -> Optional[CoverEffectInfo]:
        """
        Extract metadata from an effect class.

        Expected class attributes on the subclass:
          - effect_id (str)
          - effect_name (str)
          - effect_description (str)
          - effect_author (str)
          - effect_version (str)
          - effect_max_inputs (int)
          - @classmethod parameters() -> Dict[str, PluginParameter]
        """
        # effect_id is mandatory: if missing, we skip the effect
        effect_id = getattr(cls, "effect_id", None)
        if not effect_id or not isinstance(effect_id, str):
            return None

        name = getattr(cls, "effect_name", effect_id)
        description = getattr(cls, "effect_description", "") or ""
        author = getattr(cls, "effect_author", "Unknown")
        version = getattr(cls, "effect_version", "1.0")
        max_inputs = int(getattr(cls, "effect_max_inputs", 1))

        parameters: Dict[str, PluginParameter] = {}
        try:
            if hasattr(cls, "parameters") and callable(getattr(cls, "parameters")):
                parameters = cls.parameters()  # type: ignore[assignment]
        except Exception:
            # If parameters() fails, we just expose an empty schema
            parameters = {}

        return CoverEffectInfo(
            effect_id=effect_id,
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
    # Public API used by the 2D tab                                      #
    # ------------------------------------------------------------------ #

    def list_plugins(self) -> List[CoverEffectInfo]:
        """
        Return a list of discovered effects.

        Name is kept as list_plugins() for symmetry with the 3D manager so that
        the CoverVisualizationsTab can call the same method.
        """
        return list(self._effects.values())

    def get_plugin(self, effect_id: str) -> Optional[CoverEffectInfo]:
        """
        Return the metadata for a single effect by its ID.
        """
        return self._effects.get(effect_id)

    def create_instance(
        self,
        effect_id: str,
        config: Optional[Dict[str, object]] = None,
    ) -> Optional[BaseCoverEffect]:
        """
        Instantiate a cover effect class from its effect_id.

        Returns an instance of BaseCoverEffect (or subclass), or None
        if anything goes wrong.
        """
        info = self._effects.get(effect_id)
        if info is None:
            return None

        try:
            # Reload the module if it is not currently loaded
            module = sys.modules.get(info.module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(
                    info.module_name,
                    info.module_path,
                )
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules[info.module_name] = module
                spec.loader.exec_module(module)  # type: ignore[arg-type]

            cls_obj = getattr(module, info.class_name, None)
            if cls_obj is None:
                return None
            if not issubclass(cls_obj, BaseCoverEffect):
                return None

            return cls_obj(config=config or {})
        except Exception:
            # Never crash the host because of a broken effect
            return None
