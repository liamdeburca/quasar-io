from typing import Self

from quasar_decrypt.utils._specwindow import _SpecWindow

from ..core import BaseGroup
from .spec import XBoundsDataset
from .utils import ClassAndVersionAttrs


class SpecWindowGroup(BaseGroup):
    attrs: ClassAndVersionAttrs
    x_bounds: XBoundsDataset


    @classmethod
    def _fromSpecWindow(cls, value: _SpecWindow) -> dict:
        from numpy import float64
        return {
            "attrs": {"class_name": value.__class__.__name__},
            "x_bounds": {
                "data": value.x_bounds,
                "dtype": float64,
            }
        }


    @classmethod
    def fromSpecWindow(cls, value: _SpecWindow) -> Self:
        return cls(**cls._fromSpecWindow(value))


    def toSpecWindow(self, spec_window: _SpecWindow) -> None:
        if self.x_bounds is None:
            raise ValueError(
                "'x_bounds' is missing in SpecWindowGroup."
            )
        spec_window.x_bounds = tuple(map(float, self.x_bounds.data))
