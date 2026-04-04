"""Configuration loader for RAG system."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_path: str | None = None) -> None:
        """
        Initialize ConfigLoader.

        Args:
            config_path: Path to config.yaml. If None, looks in project root.
        """
        if config_path is None:
            # Look for config at project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            config_path = str(project_root / "config" / "config.yaml")

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.config = self._load_yaml()
        logger.info(f"Loaded config from {self.config_path}")

    def _load_yaml(self) -> dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            path: Dot-separated path (e.g., "embedding.provider")
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        parts = path.split(".")
        current = self.config

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return default
            else:
                return default

        return current

    def get_section(self, section: str) -> dict[str, Any]:
        """Get a full section of configuration."""
        return self.config.get(section, {})

    def to_dict(self) -> dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self.config.copy()
