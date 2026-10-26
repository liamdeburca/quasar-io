"""
Utilities for handling different serialization formats:
- JSON (default)
- YAML
- TOML
- TOON
"""
from importlib import import_module
from types import ModuleType
from typing import Literal

from pydantic import validate_call
from quasar_typing.literals import JSONExt


@validate_call
def load_pkg(
    suffix: JSONExt,
) -> tuple[
    Literal["json", "yaml", "toml", "toons"], 
    ModuleType,
]:
    """
    Load the package corresponding to the given file suffix. Returns a tuple of 
    the package name and the imported module.

    The options are:
    - ".json" for the JSON package
    - ".yaml" or ".yml" for the YAML package
    - ".toml" for the TOML package
    - ".toon" for the TOON package

    Parameters
    ----------
    suffix : JSONExt
        The file suffix indicating the serialization format.

    Returns
    -------
    Literal["json", "yaml", "toml", "toons"]
        The name of the package corresponding to the given file suffix.
    ModuleType
        The imported module for the package.
    """
    match suffix:
        case ".json":
            pkg_name = "json"
        case ".yaml" | ".yml":
            pkg_name = "yaml"
        case ".toml":
            pkg_name = "toml"
        case ".toon":
            pkg_name = "toons"
        case _:
            raise ValueError(f"Unsupported suffix: {suffix}")
        
    try:
        return (pkg_name, import_module(pkg_name))
    except ImportError as e:
        raise ImportError(
            f"Failed to import package '{pkg_name}' for suffix '{suffix}'"
        ) from e