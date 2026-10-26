from typing import ClassVar, Literal, Self

from quasar_decrypt import Spectrum
from quasar_decrypt.iron import IronWindows, IWindow

from ..core import BaseGroup
from .specwindow import SpecWindowGroup
from .utils import ClassAndVersionAttrs


class IWindowAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for IWindow class."""
    class_name: Literal["IWindow"] = "IWindow"


class IronWindowsAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for IronWindows class."""
    class_name: Literal["IronWindows"] = "IronWindows"


### IWindow


class IWindowGroup(SpecWindowGroup):
    """Basic HDF5 IWindow group representation."""
    attrs: IWindowAttrs
    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromIWindow(cls, value: IWindow) -> dict:
        return cls._fromSpecWindow(value)


    @classmethod
    def fromIWindow(cls, value: IWindow) -> Self:
        return cls(**cls._fromIWindow(value))


    def toIWindow(self, iwindow: IWindow) -> None:
        self.toSpecWindow(iwindow)


### IronWindows


class IronWindowsGroup(BaseGroup):
    """HDF5 representation of an IronWindows instance. 
    
    This class currently supports up to 10 IWindowGroup instances.
    """
    attrs: IronWindowsAttrs
    
    iwindow_0: IWindowGroup | None = None
    iwindow_1: IWindowGroup | None = None
    iwindow_2: IWindowGroup | None = None
    iwindow_3: IWindowGroup | None = None
    iwindow_4: IWindowGroup | None = None
    iwindow_5: IWindowGroup | None = None
    iwindow_6: IWindowGroup | None = None
    iwindow_7: IWindowGroup | None = None
    iwindow_8: IWindowGroup | None = None
    iwindow_9: IWindowGroup | None = None

    _is_required: ClassVar[Literal[False]] = False

    @classmethod
    def _fromIronWindows(cls, value: IronWindows) -> dict:
        data = {"attrs": {"class_name": "IronWindows"}}
        for i, iwindow in enumerate(value):
            data[f"iwindow_{i}"] = IWindowGroup._fromIWindow(iwindow)
        return data

    @classmethod
    def fromIronWindows(cls, value: IronWindows) -> Self:
        return cls(**cls._fromIronWindows(value))


    def toIronWindows(self, iron_windows: IronWindows) -> None:
        for i, iwindow in enumerate(iron_windows):
            iwindow_group: IWindowGroup | None = getattr(self, f"iwindow_{i}")
            if iwindow_group is None:
                raise ValueError(
                    f"'iwindow_{i}' is missing in IronWindowsGroup."
                )
            iwindow_group.toIWindow(iwindow)


    def toSpectrum(self, spectrum: Spectrum) -> None:
        windows = []
        i = 0
        while True:
            iwindow_group: IWindowGroup | None = getattr(self, f"iwindow_{i}")
            if iwindow_group is None:
                break
            windows.append(tuple(map(float, iwindow_group.x_bounds.data)))
            i += 1

        spectrum.instantiateIron(windows=windows)
        self.toIronWindows(spectrum.fe)