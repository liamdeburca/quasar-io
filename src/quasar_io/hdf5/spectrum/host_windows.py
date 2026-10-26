from typing import ClassVar, Literal, Self

from quasar_decrypt import Spectrum
from quasar_decrypt.host import HostWindows, HWindow

from ..core import BaseGroup
from .specwindow import SpecWindowGroup
from .utils import ClassAndVersionAttrs


class HWindowAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for HWindow class."""
    class_name: Literal["HWindow"] = "HWindow"


class HostWindowsAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for HostWindows class."""
    class_name: Literal["HostWindows"] = "HostWindows"


### HWindow


class HWindowGroup(SpecWindowGroup):
    """Basic HDF5 Hwindow group representation."""
    attrs: HWindowAttrs
    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromHWindow(cls, value: HWindow) -> dict:
        return cls._fromSpecWindow(value)


    @classmethod
    def fromHWindow(cls, value: HWindow) -> Self:
        return cls(**cls._fromHWindow(value))


    def toHWindow(self, hwindow: HWindow) -> None:
        self.toSpecWindow(hwindow)


### HostWindows


class HostWindowsGroup(BaseGroup):
    """HDF5 representation of a HostWindows instance. 
    
    This class currently supports up to 10 HWindowGroup instances.
    """
    attrs: HostWindowsAttrs
    
    hwindow_0: HWindowGroup | None = None
    hwindow_1: HWindowGroup | None = None
    hwindow_2: HWindowGroup | None = None
    hwindow_3: HWindowGroup | None = None
    hwindow_4: HWindowGroup | None = None
    hwindow_5: HWindowGroup | None = None
    hwindow_6: HWindowGroup | None = None
    hwindow_7: HWindowGroup | None = None
    hwindow_8: HWindowGroup | None = None
    hwindow_9: HWindowGroup | None = None

    _is_required: ClassVar[Literal[False]] = False

    @classmethod
    def _fromHostWindows(cls, value: HostWindows) -> dict:
        data = {"attrs": {"class_name": "HostWindows"}}
        for i, hwindow in enumerate(value):
            data[f"hwindow_{i}"] = HWindowGroup._fromHWindow(hwindow)
        return data

    @classmethod
    def fromHostWindows(cls, value: HostWindows) -> Self:
        return cls(**cls._fromHostWindows(value))


    def toHostWindows(self, host_windows: HostWindows) -> None:
        for i, hwindow in enumerate(host_windows):
            hwindow_group: HWindowGroup | None = getattr(self, f"hwindow_{i}")
            if hwindow_group is None:
                raise ValueError(
                    f"'hwindow_{i}' is missing in HostWindowsGroup."
                )
            hwindow_group.toHWindow(hwindow)

    def toSpectrum(self, spectrum: Spectrum) -> None:
        windows = []
        i = 0
        while True:
            hwindow_group: HWindowGroup | None = getattr(self, f"hwindow_{i}")
            if hwindow_group is None:
                break
            windows.append(tuple(map(float, hwindow_group.x_bounds.data)))
            i += 1

        spectrum.instantiateHost(windows=windows)
        self.toHostWindows(spectrum.hg)