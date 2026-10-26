from typing import Literal

from ..core import BaseAttrs


class ClassAndVersionAttrs(BaseAttrs):
    """Simple attributes manager for storing class name and version."""
    class_name: str
    version: Literal[1] = 1