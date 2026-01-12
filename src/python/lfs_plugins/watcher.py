# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hot reload file watcher."""

import threading
import time
from typing import Set, TYPE_CHECKING

from .plugin import PluginState

if TYPE_CHECKING:
    from .manager import PluginManager


class PluginWatcher:
    """Watch plugin files for changes and trigger reloads."""

    def __init__(self, manager: "PluginManager", poll_interval: float = 1.0):
        self.manager = manager
        self.poll_interval = poll_interval
        self._running = False
        self._thread: threading.Thread = None
        self._pending_reloads: Set[str] = set()
        self._lock = threading.Lock()

    def start(self):
        """Start the file watcher thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the file watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _watch_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                self._check_for_changes()
                self._process_pending_reloads()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def _check_for_changes(self):
        """Check all loaded plugins for file changes."""
        for name, plugin in self.manager._plugins.items():
            if plugin.state != PluginState.ACTIVE:
                continue
            if not plugin.info.hot_reload:
                continue

            if self._has_changes(plugin):
                with self._lock:
                    self._pending_reloads.add(name)

    def _has_changes(self, plugin) -> bool:
        """Check if any plugin files were modified."""
        for py_file in plugin.info.path.rglob("*.py"):
            if ".venv" in py_file.parts:
                continue

            try:
                current_mtime = py_file.stat().st_mtime
                prev_mtime = plugin.file_mtimes.get(py_file, 0)

                if current_mtime > prev_mtime:
                    return True
            except OSError:
                continue

        return False

    def _process_pending_reloads(self):
        """Process queued plugin reloads."""
        with self._lock:
            pending = self._pending_reloads.copy()
            self._pending_reloads.clear()

        for name in pending:
            try:
                self.manager.reload(name)
            except Exception:
                pass
