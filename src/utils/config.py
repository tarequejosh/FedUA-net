"""
Configuration loading utilities.
"""

import os
from typing import Any, Dict
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads YAML configuration file into a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
