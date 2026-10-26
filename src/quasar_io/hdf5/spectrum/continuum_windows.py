from typing import ClassVar, Literal, Self

from quasar_decrypt import Spectrum
from quasar_decrypt.continuum import ContinuumWindows, CWindow

from ..core import BaseGroup
from .specwindow import SpecWindowGroup
from .utils import ClassAndVersionAttrs


class CWindowAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for CWindow class."""
    class_name: Literal["CWindow"] = "CWindow"


class ContinuumWindowsAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for ContinuumWindows class."""
    class_name: Literal["ContinuumWindows"] = "ContinuumWindows"


### CWindow


class CWindowGroup(SpecWindowGroup):
    """Basic HDF5 Cwindow group representation."""
    attrs: CWindowAttrs
    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromCWindow(cls, value: CWindow) -> dict:
        return cls._fromSpecWindow(value)

    @classmethod
    def fromCWindow(cls, value: CWindow) -> Self:
        return cls(**cls._fromCWindow(value))


    def toCWindow(self, cwindow: CWindow) -> None:
        self.toSpecWindow(cwindow)


### ContinuumWindows


class ContinuumWindowsGroup(BaseGroup):
    """HDF5 representation of a ContinuumWindows instance. 
    
    This class currently supports up to 10 CWindowGroup instances.
    """
    attrs: ContinuumWindowsAttrs
    
    cwindow_0: CWindowGroup | None = None
    cwindow_1: CWindowGroup | None = None
    cwindow_2: CWindowGroup | None = None
    cwindow_3: CWindowGroup | None = None
    cwindow_4: CWindowGroup | None = None
    cwindow_5: CWindowGroup | None = None
    cwindow_6: CWindowGroup | None = None
    cwindow_7: CWindowGroup | None = None
    cwindow_8: CWindowGroup | None = None
    cwindow_9: CWindowGroup | None = None

    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromContinuumWindows(cls, value: ContinuumWindows) -> dict:
        data = {"attrs": {"class_name": "ContinuumWindows"}}
        for i, cwindow in enumerate(value):
            data[f"cwindow_{i}"] = CWindowGroup._fromCWindow(cwindow)
        return data


    @classmethod
    def fromContinuumWindows(cls, value: ContinuumWindows) -> Self:
        return cls(**cls._fromContinuumWindows(value))


    def toContinuumWindows(self, continuum_windows: ContinuumWindows) -> None:
        for i, cwindow in enumerate(continuum_windows):
            cwindow_group: CWindowGroup | None = getattr(self, f"cwindow_{i}")
            if cwindow_group is None:
                raise ValueError(
                    f"'cwindow_{i}' is missing in ContinuumWindowsGroup."
                )
            cwindow_group.toCWindow(cwindow)

    def toSpectrum(self, spectrum: Spectrum) -> None:
        windows = []
        i = 0
        while True:
            cwindow_group: CWindowGroup | None = getattr(self, f"cwindow_{i}")
            if cwindow_group is None:
                break
            windows.append(tuple(map(float, cwindow_group.x_bounds.data)))
            i += 1

        spectrum.instantiateContinuum(windows=windows)
        self.toContinuumWindows(spectrum.pl)
