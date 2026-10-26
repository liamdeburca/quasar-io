from pathlib import Path
from typing import Self

from quasar_decrypt import Spectrum
from quasar_utils.setup import Info

from quasar_io.hdf5.core import UnitHashAttrs

from .balmer_windows import BalmerWindowsGroup
from .continuum_windows import ContinuumWindowsGroup
from .host_windows import HostWindowsGroup
from .iron_windows import IronWindowsGroup
from .line_windows import LineWindowsGroup
from .spec import SpecAttrs, SpecGroup


class SpectrumAttrs(SpecAttrs, UnitHashAttrs):
    path: str
    title: str

    @classmethod
    def _fromSpectrum(cls, value: Spectrum) -> dict:
        data = cls._fromSpec(value)
        data |= cls._fromInfo(value.info)
        data |= {
            "path": str(value.path),
            "title": value.title,
        }
        return data

    @classmethod
    def fromSpectrum(cls, value: Spectrum) -> Self:
        return cls(**cls._fromSpectrum(value))


class SpectrumGroup(SpecGroup):
    attrs: SpectrumAttrs

    pl: ContinuumWindowsGroup | None = None
    fe: IronWindowsGroup | None = None
    ba: BalmerWindowsGroup | None = None
    hg: HostWindowsGroup | None = None
    em: LineWindowsGroup | None = None

    @classmethod
    def _fromSpectrum(cls, value: Spectrum) -> dict:
        data = cls._fromSpec(value)
        data["attrs"] = SpectrumAttrs._fromSpectrum(value)
        if value.pl is not None:
            data["pl"] = ContinuumWindowsGroup._fromContinuumWindows(value.pl)
        if value.fe is not None:
            data["fe"] = IronWindowsGroup._fromIronWindows(value.fe)
        if value.ba is not None:
            data["ba"] = BalmerWindowsGroup._fromBalmerWindows(value.ba)
        if value.hg is not None:
            data["hg"] = HostWindowsGroup._fromHostWindows(value.hg)
        if value.em is not None:
            data["em"] = LineWindowsGroup._fromLineWindows(value.em)
        
        return data


    @classmethod
    def fromSpectrum(cls, value: Spectrum) -> Self:
        return cls(**cls._fromSpectrum(value))


    def toSpectrum(self, info: Info) -> Spectrum:
        """
        Creates a Spectrum instance using the stored HDF5 data.
        """
        assert self.attrs is not None, "`.attrs` is missing."
        assert self.x is not None, "`.x` is missing."

        self.attrs.validate_info_instance(info)
        kwargs = {
            "path": Path(self.attrs.path),
            "title": str(self.attrs.title),
            "info": info,
        }
        kwargs.update(self._to_spectrum_helper())
            
        if self.original is None:
            kwargs = Spectrum.get_kwargs_simple(**kwargs)
        if self.original is not None:
            kwargs.update({
                f"{key}_original": value
                for key, value in self.original._to_spectrum_helper().items()
                if key not in ("x0", "y0", "x_bounds")
            })
            kwargs = Spectrum.get_kwargs(**kwargs)

        spectrum = Spectrum(**kwargs)

        if self.pl is not None:
            self.pl.toSpectrum(spectrum)
        if self.fe is not None:
            self.fe.toSpectrum(spectrum)
        if self.ba is not None:
            self.ba.toSpectrum(spectrum)
        if self.hg is not None:
            self.hg.toSpectrum(spectrum)
        if self.em is not None:
            self.em.toSpectrum(spectrum)

        return spectrum
