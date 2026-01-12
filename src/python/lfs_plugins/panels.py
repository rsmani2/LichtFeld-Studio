# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Built-in plugin system panels."""

from typing import Optional, List
import threading

MAX_OUTPUT_LINES = 100


class PluginManagerPanel:
    """GUI panel for managing plugins."""

    panel_label = "Plugin Manager"
    panel_space = "SIDE_PANEL"
    panel_order = 99

    def __init__(self):
        self.github_url = ""
        self.status_message = ""
        self.status_is_error = False
        self.selected_plugin_idx = -1
        self._operation_in_progress = False
        self._output_lines: List[str] = []
        self._lock = threading.Lock()

    def draw(self, layout):
        from .manager import PluginManager
        from .plugin import PluginState

        mgr = PluginManager.instance()

        # Install from GitHub section
        if layout.collapsing_header("Install from GitHub", default_open=True):
            layout.label("GitHub URL or shorthand:")
            _, self.github_url = layout.input_text("##github_url", self.github_url)

            layout.spacing()

            # Show supported formats
            if layout.tree_node("Supported formats"):
                layout.bullet_text("https://github.com/owner/repo")
                layout.bullet_text("github:owner/repo")
                layout.bullet_text("owner/repo")
                layout.tree_pop()

            layout.spacing()

            with self._lock:
                in_progress = self._operation_in_progress

            if in_progress:
                layout.progress_bar(-1.0, self.status_message or "Working...")
                if self._output_lines:
                    if layout.tree_node("Output"):
                        for line in self._output_lines[-15:]:
                            layout.text_wrapped(line)
                        layout.tree_pop()
            else:
                if layout.button("Install Plugin", (0, 28)):
                    self._install_plugin(mgr)

        layout.separator()

        # Installed plugins section
        if layout.collapsing_header("Installed Plugins", default_open=True):
            plugins = mgr.discover()

            if not plugins:
                layout.text_colored("No plugins installed", (0.6, 0.6, 0.6, 1.0))
            else:
                plugin_names = [p.name for p in plugins]
                _, self.selected_plugin_idx = layout.listbox(
                    "##plugins", self.selected_plugin_idx, plugin_names, 5
                )

                if 0 <= self.selected_plugin_idx < len(plugins):
                    plugin = plugins[self.selected_plugin_idx]
                    state = mgr.get_state(plugin.name)

                    layout.spacing()
                    layout.label(f"Version: {plugin.version}")
                    if plugin.description:
                        layout.label(f"Description: {plugin.description}")

                    state_str = state.value if state else "not loaded"
                    if state == PluginState.ACTIVE:
                        layout.text_colored(f"Status: {state_str}", (0.3, 0.9, 0.3, 1.0))
                    elif state == PluginState.ERROR:
                        layout.text_colored(f"Status: {state_str}", (0.9, 0.3, 0.3, 1.0))
                        error = mgr.get_error(plugin.name)
                        if error:
                            layout.text_wrapped(error)
                        tb = mgr.get_traceback(plugin.name)
                        if tb and layout.tree_node("Traceback"):
                            for line in tb.strip().split("\n"):
                                layout.text_wrapped(line)
                            layout.tree_pop()
                    else:
                        layout.label(f"Status: {state_str}")

                    layout.spacing()

                    # Action buttons
                    with self._lock:
                        in_progress = self._operation_in_progress

                    if not in_progress:
                        if state == PluginState.ACTIVE:
                            if layout.button("Reload"):
                                self._reload_plugin(mgr, plugin.name)
                            layout.same_line()
                            if layout.button("Unload"):
                                self._unload_plugin(mgr, plugin.name)
                        else:
                            if layout.button("Load"):
                                self._load_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button("Update"):
                            self._update_plugin(mgr, plugin.name)

                        layout.same_line()
                        if layout.button("Uninstall"):
                            self._uninstall_plugin(mgr, plugin.name)

        # Status message
        if self.status_message:
            layout.separator()
            if self.status_is_error:
                layout.text_colored(self.status_message, (0.9, 0.3, 0.3, 1.0))
            else:
                layout.text_colored(self.status_message, (0.3, 0.9, 0.3, 1.0))

    def _set_status(self, message: str, is_error: bool = False):
        self.status_message = message
        self.status_is_error = is_error

    def _add_output(self, line: str):
        with self._lock:
            self._output_lines.append(line)
            if len(self._output_lines) > MAX_OUTPUT_LINES:
                self._output_lines = self._output_lines[-MAX_OUTPUT_LINES:]

    def _clear_output(self):
        with self._lock:
            self._output_lines.clear()

    def _run_async(self, operation, success_msg: str, error_prefix: str):
        def on_progress(msg: str):
            self._set_status(msg)
            self._add_output(msg)

        def worker():
            with self._lock:
                self._operation_in_progress = True
            self._clear_output()
            try:
                result = operation(on_progress)
                self._set_status(success_msg.format(result) if result else success_msg)
            except Exception as e:
                self._set_status(f"{error_prefix}: {e}", True)
            finally:
                with self._lock:
                    self._operation_in_progress = False

        threading.Thread(target=worker, daemon=True).start()

    def _install_plugin(self, mgr):
        url = self.github_url.strip()
        if not url:
            self._set_status("Please enter a GitHub URL", True)
            return

        def do_install(on_progress):
            name = mgr.install(url, on_progress=on_progress)
            self.github_url = ""
            return name

        self._run_async(do_install, "Installed: {}", "Install failed")

    def _load_plugin(self, mgr, name: str):
        self._run_async(
            lambda cb: mgr.load(name, on_progress=cb),
            f"Loaded: {name}", "Load failed"
        )

    def _unload_plugin(self, mgr, name: str):
        try:
            mgr.unload(name)
            self._set_status(f"Unloaded: {name}")
        except Exception as e:
            self._set_status(f"Unload failed: {e}", True)

    def _reload_plugin(self, mgr, name: str):
        def do_reload(on_progress):
            mgr.unload(name)
            mgr.load(name, on_progress=on_progress)

        self._run_async(do_reload, f"Reloaded: {name}", "Reload failed")

    def _update_plugin(self, mgr, name: str):
        self._run_async(
            lambda cb: mgr.update(name, on_progress=cb),
            f"Updated: {name}", "Update failed"
        )

    def _uninstall_plugin(self, mgr, name: str):
        try:
            mgr.uninstall(name)
            self._set_status(f"Uninstalled: {name}")
            self.selected_plugin_idx = -1
        except Exception as e:
            self._set_status(f"Uninstall failed: {e}", True)


def register_builtin_panels():
    """Register built-in plugin system panels."""
    try:
        import lichtfeld as lf

        lf.ui.register_panel(PluginManagerPanel)
    except ImportError:
        pass  # lichtfeld module not available (e.g., testing)
