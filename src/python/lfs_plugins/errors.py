# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plugin system exceptions."""


class PluginError(Exception):
    """Base exception for plugin errors."""

    pass


class PluginLoadError(PluginError):
    """Failed to load plugin."""

    pass


class PluginDependencyError(PluginError):
    """Failed to install dependencies."""

    pass
