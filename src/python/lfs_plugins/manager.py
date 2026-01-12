# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin Manager - discovery, loading, lifecycle."""

import sys
import threading
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Callable

from .plugin import PluginInfo, PluginInstance, PluginState
from .installer import PluginInstaller, clone_from_url, update_plugin, uninstall_plugin
from .watcher import PluginWatcher
from .errors import PluginError

try:
    import tomllib
except ImportError:
    import tomli as tomllib


class PluginManager:
    """Singleton managing plugin discovery, loading, and lifecycle."""

    _instance: Optional["PluginManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._plugins: Dict[str, PluginInstance] = {}
        self._plugins_dir = Path.home() / ".lichtfeld" / "plugins"
        self._watcher: Optional[PluginWatcher] = None
        self._on_plugin_loaded: List[Callable] = []
        self._on_plugin_unloaded: List[Callable] = []

    @classmethod
    def instance(cls) -> "PluginManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def plugins_dir(self) -> Path:
        return self._plugins_dir

    def discover(self) -> List[PluginInfo]:
        """Scan plugins directory for valid plugins."""
        plugins = []
        if not self._plugins_dir.exists():
            self._plugins_dir.mkdir(parents=True, exist_ok=True)
            return plugins

        for entry in self._plugins_dir.iterdir():
            if entry.is_dir() and (entry / "plugin.toml").exists():
                try:
                    info = self._parse_manifest(entry)
                    plugins.append(info)
                except Exception:
                    pass  # Skip invalid plugins
        return plugins

    def _parse_manifest(self, plugin_dir: Path) -> PluginInfo:
        """Parse plugin.toml manifest."""
        with open(plugin_dir / "plugin.toml", "rb") as f:
            data = tomllib.load(f)

        plugin = data.get("plugin", {})
        deps = data.get("dependencies", {})
        lifecycle = data.get("lifecycle", {})

        return PluginInfo(
            name=plugin.get("name", plugin_dir.name),
            version=plugin.get("version", "0.0.0"),
            path=plugin_dir,
            description=plugin.get("description", ""),
            author=plugin.get("author", ""),
            entry_point=plugin.get("entry_point", "__init__"),
            dependencies=deps.get("packages", []),
            auto_start=lifecycle.get("auto_start", True),
            hot_reload=lifecycle.get("hot_reload", True),
            min_lichtfeld_version=plugin.get("min_lichtfeld_version", ""),
        )

    def load(self, name: str, on_progress: Optional[Callable] = None) -> bool:
        """Load a plugin by name."""
        plugin = self._plugins.get(name)
        if not plugin:
            for info in self.discover():
                if info.name == name:
                    plugin = PluginInstance(info=info)
                    self._plugins[name] = plugin
                    break

        if not plugin:
            raise PluginError(f"Plugin '{name}' not found")

        try:
            plugin.state = PluginState.INSTALLING
            installer = PluginInstaller(plugin)
            installer.ensure_venv()
            installer.install_dependencies(on_progress)

            plugin.state = PluginState.LOADING
            self._load_module(plugin)

            if hasattr(plugin.module, "on_load"):
                plugin.module.on_load()

            plugin.state = PluginState.ACTIVE
            self._update_file_mtimes(plugin)

            for cb in self._on_plugin_loaded:
                cb(plugin.info)

            return True

        except Exception as e:
            plugin.state = PluginState.ERROR
            plugin.error = str(e)
            plugin.error_traceback = traceback.format_exc()
            return False

    def _load_module(self, plugin: PluginInstance):
        """Import plugin module with path isolation."""
        original_path = sys.path.copy()
        try:
            venv_site = self._get_venv_site_packages(plugin)
            if venv_site and venv_site.exists():
                sys.path.insert(0, str(venv_site))
            sys.path.insert(0, str(plugin.info.path))

            entry_file = plugin.info.path / f"{plugin.info.entry_point}.py"
            spec = importlib.util.spec_from_file_location(plugin.info.name, entry_file)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[plugin.info.name] = module
            spec.loader.exec_module(module)
            plugin.module = module

        finally:
            sys.path[:] = original_path

    def _get_venv_site_packages(self, plugin: PluginInstance) -> Optional[Path]:
        """Get site-packages path for plugin venv."""
        venv = plugin.venv_path
        if not venv or not venv.exists():
            return None

        lib_dir = venv / "lib"
        if lib_dir.exists():
            for d in lib_dir.iterdir():
                if d.name.startswith("python"):
                    sp = d / "site-packages"
                    if sp.exists():
                        return sp

        # Windows layout
        sp = venv / "Lib" / "site-packages"
        if sp.exists():
            return sp

        return None

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        plugin = self._plugins.get(name)
        if not plugin or plugin.state != PluginState.ACTIVE:
            return False

        try:
            if plugin.module and hasattr(plugin.module, "on_unload"):
                plugin.module.on_unload()

            if plugin.info.name in sys.modules:
                del sys.modules[plugin.info.name]

            plugin.module = None
            plugin.state = PluginState.UNLOADED

            for cb in self._on_plugin_unloaded:
                cb(plugin.info)

            return True

        except Exception as e:
            plugin.error = str(e)
            plugin.state = PluginState.UNLOADED
            return False

    def reload(self, name: str) -> bool:
        """Hot reload a plugin."""
        self.unload(name)
        return self.load(name)

    def load_all(self) -> Dict[str, bool]:
        """Load all discovered plugins with auto_start=True."""
        results = {}
        for info in self.discover():
            if info.auto_start:
                results[info.name] = self.load(info.name)
        return results

    def list_loaded(self) -> List[str]:
        """List names of loaded plugins."""
        return [
            name for name, p in self._plugins.items() if p.state == PluginState.ACTIVE
        ]

    def get_info(self, name: str) -> Optional[PluginInfo]:
        """Get plugin info by name."""
        plugin = self._plugins.get(name)
        return plugin.info if plugin else None

    def get_state(self, name: str) -> Optional[PluginState]:
        """Get plugin state by name."""
        plugin = self._plugins.get(name)
        return plugin.state if plugin else None

    def get_error(self, name: str) -> Optional[str]:
        """Get plugin error message if any."""
        plugin = self._plugins.get(name)
        return plugin.error if plugin else None

    def get_traceback(self, name: str) -> Optional[str]:
        """Get plugin error traceback if any."""
        plugin = self._plugins.get(name)
        return plugin.error_traceback if plugin else None

    def _update_file_mtimes(self, plugin: PluginInstance):
        """Record file modification times for hot reload."""
        plugin.file_mtimes.clear()
        for py_file in plugin.info.path.rglob("*.py"):
            if ".venv" not in py_file.parts:
                plugin.file_mtimes[py_file] = py_file.stat().st_mtime

    def start_watcher(self, poll_interval: float = 1.0):
        """Start hot reload file watcher."""
        if self._watcher:
            return
        self._watcher = PluginWatcher(self, poll_interval)
        self._watcher.start()

    def stop_watcher(self):
        """Stop hot reload file watcher."""
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def on_plugin_loaded(self, callback: Callable):
        """Register callback for plugin load events."""
        self._on_plugin_loaded.append(callback)

    def on_plugin_unloaded(self, callback: Callable):
        """Register callback for plugin unload events."""
        self._on_plugin_unloaded.append(callback)

    def install(
        self,
        url: str,
        on_progress: Optional[Callable[[str], None]] = None,
        auto_load: bool = True,
    ) -> str:
        """Install a plugin from GitHub URL.

        Args:
            url: GitHub URL or shorthand (github:owner/repo, owner/repo)
            on_progress: Optional progress callback
            auto_load: Whether to load the plugin after installation

        Returns:
            Name of the installed plugin
        """
        plugin_dir = clone_from_url(url, self._plugins_dir, on_progress)
        info = self._parse_manifest(plugin_dir)

        if auto_load:
            self.load(info.name, on_progress)

        return info.name

    def update(
        self,
        name: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """Update a plugin by pulling latest from git.

        Args:
            name: Plugin name
            on_progress: Optional progress callback

        Returns:
            True if updated successfully
        """
        plugin = self._plugins.get(name)
        if plugin:
            plugin_dir = plugin.info.path
        else:
            # Try to find by discovering plugins
            for info in self.discover():
                if info.name == name:
                    plugin_dir = info.path
                    break
            else:
                raise PluginError(f"Plugin '{name}' not found")

        was_loaded = plugin and plugin.state == PluginState.ACTIVE
        if was_loaded:
            self.unload(name)

        update_plugin(plugin_dir, on_progress)

        if was_loaded:
            self.load(name, on_progress)

        return True

    def uninstall(self, name: str) -> bool:
        """Uninstall a plugin by removing its directory.

        Args:
            name: Plugin name

        Returns:
            True if uninstalled successfully
        """
        plugin = self._plugins.get(name)
        if plugin:
            if plugin.state == PluginState.ACTIVE:
                self.unload(name)
            plugin_dir = plugin.info.path
            del self._plugins[name]
        else:
            # Try to find by discovering plugins
            for info in self.discover():
                if info.name == name:
                    plugin_dir = info.path
                    break
            else:
                raise PluginError(f"Plugin '{name}' not found")

        return uninstall_plugin(plugin_dir)
