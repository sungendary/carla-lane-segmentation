"""
Configuration loader.

Reads configs/config.yaml and exposes it as a nested dict. If config.yaml
does not exist, it raises a clear error telling the user to copy the
example file. This keeps actual values out of git while documenting the
expected structure in config.example.yaml.
"""
import os
import yaml

# repo root = one level up from this file's folder (src/)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, 'configs', 'config.yaml')
_EXAMPLE_PATH = os.path.join(_REPO_ROOT, 'configs', 'config.example.yaml')


def load_config(section=None):
    """Load the YAML config. If `section` is given (e.g. 'train'), return
    just that sub-dict; otherwise return the whole config."""
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(
            f"Config not found at {_CONFIG_PATH}.\n"
            f"Copy the example and edit it:\n"
            f"    cp {_EXAMPLE_PATH} {_CONFIG_PATH}"
        )
    with open(_CONFIG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    if section is not None:
        if section not in cfg:
            raise KeyError(f"Section '{section}' not found in config.yaml")
        return cfg[section]
    return cfg


def repo_path(*parts):
    """Build a path relative to the repo root, so scripts work regardless
    of the current working directory."""
    return os.path.join(_REPO_ROOT, *parts)
