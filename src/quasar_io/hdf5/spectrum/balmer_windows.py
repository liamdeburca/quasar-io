from typing import ClassVar, Literal, Self

from quasar_decrypt import Spectrum
from quasar_decrypt.balmer import BalmerWindows, BWindow

from ..core import BaseGroup
from .specwindow import SpecWindowGroup
from .utils import ClassAndVersionAttrs


class BWindowAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for BWindow class."""
    class_name: Literal["BWindow"] = "BWindow"


class BalmerWindowsAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for BalmerWindows class."""
    class_name: Literal["BalmerWindows"] = "BalmerWindows"


### BWindow


class BWindowGroup(SpecWindowGroup):
    """Basic HDF5 Bwindow group representation."""
    attrs: BWindowAttrs
    _is_required: ClassVar[Literal[False]] = False


    @classmethod
    def _fromBWindow(cls, value: BWindow) -> dict:
        return cls._fromSpecWindow(value)

    @classmethod
    def fromBWindow(cls, value: BWindow) -> Self:
        return cls(**cls._fromBWindow(value))


    def toBWindow(self, bwindow: BWindow) -> None:
        self.toSpecWindow(bwindow)


### BalmerWindows


class BalmerWindowsGroup(BaseGroup):
    """HDF5 representation of a BalmerWindows instance. 
    
    This class currently supports up to 10 BWindowGroup instances.
    """
    attrs: BalmerWindowsAttrs
    
    bwindow_0: BWindowGroup | None = None
    bwindow_1: BWindowGroup | None = None
    bwindow_2: BWindowGroup | None = None
    bwindow_3: BWindowGroup | None = None
    bwindow_4: BWindowGroup | None = None
    bwindow_5: BWindowGroup | None = None
    bwindow_6: BWindowGroup | None = None
    bwindow_7: BWindowGroup | None = None
    bwindow_8: BWindowGroup | None = None
    bwindow_9: BWindowGroup | None = None

    _is_required: ClassVar[Literal[False]] = False

    @classmethod
    def _fromBalmerWindows(cls, value: BalmerWindows) -> dict:
        data = {"attrs": {"class_name": "BalmerWindows"}}
        for i, bwindow in enumerate(value):
            data[f"bwindow_{i}"] = BWindowGroup._fromBWindow(bwindow)
        return data


    @classmethod
    def fromBalmerWindows(cls, value: BalmerWindows) -> Self:
        return cls(**cls._fromBalmerWindows(value))


    def toBalmerWindows(self, balmer_windows: BalmerWindows) -> None:
        for i, bwindow in enumerate(balmer_windows):
            bwindow_group: BWindowGroup | None = getattr(self, f"bwindow_{i}")
            if bwindow_group is None:
                raise ValueError(
                    f"'bwindow_{i}' is missing in BalmerWindowsGroup."
                )
            bwindow_group.toBWindow(bwindow)

    def toSpectrum(self, spectrum: Spectrum) -> None:
        windows = []
        i = 0
        while True:
            bwindow_group: BWindowGroup | None = getattr(self, f"bwindow_{i}")
            if bwindow_group is None:
                break
            windows.append(tuple(map(float, bwindow_group.x_bounds.data)))
            i += 1

        spectrum.instantiateBalmer(windows=windows)
        self.toBalmerWindows(spectrum.ba)
        
