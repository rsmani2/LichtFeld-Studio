# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""LichtFeld Plugin System."""

from .manager import PluginManager
from .plugin import PluginInfo, PluginState, PluginInstance
from .errors import PluginError, PluginLoadError, PluginDependencyError
from .panels import PluginManagerPanel, register_builtin_panels

__all__ = [
    "PluginManager",
    "PluginInfo",
    "PluginState",
    "PluginInstance",
    "PluginError",
    "PluginLoadError",
    "PluginDependencyError",
    "PluginManagerPanel",
    "register_builtin_panels",
]
