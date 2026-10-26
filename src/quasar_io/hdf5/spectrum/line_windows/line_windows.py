from typing import ClassVar, Literal, Self

from quasar_decrypt import Spectrum
from quasar_decrypt.lines import LineWindows, LWindow

from ... import core
from ..utils import ClassAndVersionAttrs
from .lwindow import LWindowGroup


class LineWindowsAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for LineWindows class."""
    class_name: Literal["LineWindows"]
    n_windows: int


    @classmethod
    def _fromLineWindows(cls, value: LineWindows) -> dict:
        return {
            "class_name": value.__class__.__name__,
            "n_windows": len(value),
        }


    @classmethod
    def fromLineWindows(cls, value: LineWindows) -> Self:
        return cls(**cls._fromLineWindows(value))


### LineWindows


class LineWindowsGroup(core.BaseGroup):
    """HDF5 representation of a LineWindows instance. 
    
    This class currently supports up to 10 LWindowGroup instances.
    """
    attrs: LineWindowsAttrs
    
    lwindow_0: LWindowGroup | None = None
    lwindow_1: LWindowGroup | None = None
    lwindow_2: LWindowGroup | None = None
    lwindow_3: LWindowGroup | None = None
    lwindow_4: LWindowGroup | None = None
    lwindow_5: LWindowGroup | None = None
    lwindow_6: LWindowGroup | None = None
    lwindow_7: LWindowGroup | None = None
    lwindow_8: LWindowGroup | None = None
    lwindow_9: LWindowGroup | None = None

    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromLineWindows(cls, value: LineWindows) -> dict:
        data = {
            f"lwindow_{i}": LWindowGroup._fromLWindow(lwindow)
            for i, lwindow in enumerate(value)
        }
        data["attrs"] = LineWindowsAttrs._fromLineWindows(value)
        return data

    
    @classmethod
    def fromLineWindows(cls, value: LineWindows) -> Self:
        return cls(**cls._fromLineWindows(value))


    def toLineWindows(self, line_windows: LineWindows) -> None:
        for i in range(self.attrs.n_windows):
            lwindow_group: LWindowGroup | None = getattr(self, f"lwindow_{i}")
            if lwindow_group is None:
                break

            lwindow = LWindow.create(
                spectrum=line_windows.spectrum,
                x_bounds=tuple(map(float, lwindow_group.x_bounds.data)),
            )
            lwindow_group.toLWindow(lwindow)
            line_windows.append(lwindow)

    def toSpectrum(self, spectrum: Spectrum) -> None:
        spectrum.instantiateLines()
        self.toLineWindows(spectrum.em)
