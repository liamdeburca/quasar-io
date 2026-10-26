import math
from typing import ClassVar, Literal, Self

import pydantic
from quasar_decrypt.lines import LWindow

from ... import core
from ..specwindow import SpecWindowGroup
from ..utils import ClassAndVersionAttrs
from .utils import clear_lwindow


class LWindowAttrs(ClassAndVersionAttrs):
    """Simple attributes manager for LWindow class."""
    class_name: Literal["LWindow"] = "LWindow"


### LWindow


class _NeighboursDataset(core.IntegerVectorDataset):
    shape: tuple[Literal[2]] = (2,)


class _FloatBoundsDataset(core.FloatMatrixDataset):
    shape: tuple[int, Literal[2]] | None = None


class _CopiesToGroup(core.BaseGroup):
    _is_required: ClassVar[Literal[True]] = True

    counts: core.IntegerVectorDataset | None = None
    to_idx: core.IntegerVectorDataset | None = None
    to_name: core.StringVectorDataset | None = None


    @pydantic.model_validator(mode="after")
    def validate_dataset_shapes(self) -> Self:
        if self.counts is not None:
            n = len(self.counts.data)
            m = len(self.to_idx.data)
            o = len(self.to_name.data)
            if m != o:
                raise ValueError(
                    "Length mismatch between fields 'counts' and 'to_name': "
                    f"{n} != {o}."
                )
            if m > n:
                raise ValueError(
                    f"Length of field 'to_idx' is {m}, "
                    f"but must be <= {n}."
                )
        return self


class LWindowGroup(SpecWindowGroup):
    """Basic HDF5 LWindow group representation."""
    attrs: LWindowAttrs

    names: core.StringVectorDataset | None = None
    complexes: core.StringVectorDataset | None = None
    linetypes: core.StringVectorDataset | None = None

    waves: core.FloatVectorDataset | None = None
    n_maxs: core.IntegerVectorDataset | None = None
    
    needs_line: core.StringVectorDataset | None = None
    
    scale_init: core.FloatVectorDataset | None = None
    scale_fixed: core.BooleanVectorDataset | None = None

    is_copy_of: core.StringVectorDataset | None = None
    is_multiplet_of: core.StringVectorDataset | None = None

    blacklist: core.BooleanVectorDataset | None = None
    cropped: core.BooleanVectorDataset | None = None

    i_bounds: _FloatBoundsDataset | None = None
    strength_bounds: _FloatBoundsDataset | None = None
    fwhm_v_bounds: _FloatBoundsDataset | None = None
    v_off_bounds: _FloatBoundsDataset | None = None
    scale_bounds: _FloatBoundsDataset | None = None

    copies_to: _CopiesToGroup | None = None

    neighbours: _NeighboursDataset | None = None

    _is_required: ClassVar[Literal[False]] = False


    @pydantic.model_validator(mode="after")
    def validate_dataset_shapes(self) -> Self:
        if self.names is not None:
            n = self.names.shape[0]
            for field_name in (
                "complexes", "linetypes", "waves", "n_maxs", "needs_line", 
                "scale_init", "scale_fixed", "is_copy_of", "is_multiplet_of", 
                "blacklist", "cropped", 
                "i_bounds", "strength_bounds", "fwhm_v_bounds", "v_off_bounds", 
                "scale_bounds",
            ):
                field_value = getattr(self, field_name)
                if field_value is None:
                    raise ValueError(
                        f"Dataset '{field_name}' cannot be missing if `names` "
                        "is not."
                        )
                m = field_value.shape[0]
                if m != n:
                    raise ValueError(
                        f"Axis 0 of dataset '{field_name}' has length {m}, "
                        f"but expected {n}."
                    )
        return self


    ### Factory methods


    @classmethod
    def _fromLWindow(cls, value: LWindow) -> dict:
        def f(bounds: tuple[float | None, float | None]) -> tuple[float, float]:
            lb, ub = bounds
            if lb is None:
                lb = -math.inf
            if ub is None:
                ub = math.inf
            return lb, ub

        data = cls._fromSpecWindow(value)

        data["names"] = core.StringVectorDataset._fromData(
            names := list(value.names),
            str,
        )
        data["complexes"] = core.StringVectorDataset._fromData(
            complexes := [value.complexes[n] for n in names],
            str,
        )
        data["linetypes"] = core.StringVectorDataset._fromData(
            [value.linetypes[n] for n in names],
            str,
        )
        data["waves"] = core.FloatVectorDataset._fromData(
            [value.waves[n] for n in names],
            float,
        )
        data["n_maxs"] = core.IntegerVectorDataset._fromData(
            [value.n_maxs[n] for n in names],
            int,
        )
        data["needs_line"] = core.StringVectorDataset._fromData(
            [value.needs_line.get(n, "") for n in names],
            str,
        )
        data["scale_init"] = core.FloatVectorDataset._fromData(
            [value.scale_init.get(n, math.nan) for n in names],
            float,
        )
        data["scale_fixed"] = core.BooleanVectorDataset._fromData(
            [value.scale_fixed.get(n, False) for n in names],
            bool,
        )
        data["is_copy_of"] = core.StringVectorDataset._fromData(
            [value.is_copy_of.get(n, "") for n in names],
            str,
        )
        data["is_multiplet_of"] = core.StringVectorDataset._fromData(
            [value.is_multiplet_of.get(n, "") for n in names],
            str,
        )
        data["blacklist"] = core.BooleanVectorDataset._fromData(
            [value.blacklist.get(n, False) for n in names],
            bool,
        )
        data["cropped"] = core.BooleanVectorDataset._fromData(
            [(n in value.cropped) for n in names],
            bool,
        )
        data["i_bounds"] = _FloatBoundsDataset._fromData(
            [value.i_bounds[c] for c in complexes],
            float,
        )
        data["strength_bounds"] = _FloatBoundsDataset._fromData(
            [f(value.strength_bounds[n]) for n in names],
            float,
        )
        data["fwhm_v_bounds"] = _FloatBoundsDataset._fromData(
            [f(value.fwhm_v_bounds[n]) for n in names],
            float,
        )
        data["v_off_bounds"] = _FloatBoundsDataset._fromData(
            [f(value.v_off_bounds[n]) for n in names],
            float,
        )
        data["scale_bounds"] = _FloatBoundsDataset._fromData(
            [f(value.scale_bounds.get(n, (None, None))) for n in names],
            float,
        )

        copies_to_counts = []
        copies_to_to_idx = []
        copies_to_to_name = []
        for name in names:
            lst: list[tuple[int | None, str]] = value.copies_to.get(name, [])
            copies_to_counts.append(len(lst))
            for idx, mimic in lst:
                copies_to_to_idx.append(int(-1 if idx is None else idx))
                copies_to_to_name.append(str(mimic))

        data["copies_to"] = {
            "counts": core.IntegerVectorDataset._fromData(copies_to_counts, int),
            "to_idx": core.IntegerVectorDataset._fromData(copies_to_to_idx, int),
            "to_name": core.StringVectorDataset._fromData(copies_to_to_name, str),
        }
        data["neighbours"] = _NeighboursDataset._fromData(
            [-1 if idx is None else int(idx) for idx in value.neighbours],
            int,
        )
        data["attrs"] = LWindowAttrs().model_dump()
        return data


    @classmethod
    def fromLWindow(cls, value: LWindow) -> Self:
        data = cls._fromLWindow(value)
        return cls(**data)


    ###

    
    def toLWindow(self, lwindow: LWindow) -> None:
        clear_lwindow(lwindow)

        names = self.names.data
        complexes = self.complexes.data
        linetypes = self.linetypes.data
        waves = self.waves.data
        n_maxs = self.n_maxs.data
        needs_line = self.needs_line.data
        scale_init = self.scale_init.data
        scale_fixed = self.scale_fixed.data
        is_copy_of = self.is_copy_of.data
        is_multiplet_of = self.is_multiplet_of.data
        blacklist = self.blacklist.data
        cropped = self.cropped.data

        i_bounds = self.i_bounds.data
        strength_bounds = self.strength_bounds.data
        fwhm_v_bounds = self.fwhm_v_bounds.data
        v_off_bounds = self.v_off_bounds.data
        scale_bounds = self.scale_bounds.data

        for i, name in enumerate(names):
            lwindow.names.add(name)
        
            lwindow.linetypes[name] = str(linetypes[i])
            lwindow.waves[name] = float(waves[i])
            lwindow.n_maxs[name] = int(n_maxs[i])

            _complex = str(complexes[i])
            lwindow.complexes[name] = _complex
            if _complex not in lwindow.i_bounds:
                lwindow.i_bounds[_complex] = tuple(map(float, i_bounds[i]))


            if needs_line[i]:
                lwindow.needs_line[name] = str(needs_line[i])

            if math.isfinite(scale_init[i]):
                lwindow.scale_init[name] = float(scale_init[i])
                lwindow.scale_fixed[name] = bool(scale_fixed[i])
                lwindow.scale_bounds[name] = tuple(map(float, scale_bounds[i]))

            if is_copy_of[i]:
                lwindow.is_copy_of[name] = str(is_copy_of[i])
            if is_multiplet_of[i]:
                lwindow.is_multiplet_of[name] = str(is_multiplet_of[i])

            lwindow.blacklist[name] = bool(blacklist[i])
            lwindow._blacklist[name] = bool(blacklist[i])

            if cropped[i]:
                lwindow.cropped.add(name)

            lwindow.strength_bounds[name] = tuple(map(float, strength_bounds[i]))
            lwindow.fwhm_v_bounds[name] = tuple(map(float, fwhm_v_bounds[i]))
            lwindow.v_off_bounds[name] = tuple(map(float, v_off_bounds[i]))
        
        copies_to_counts = self.copies_to.counts.data
        copies_to_indices = self.copies_to.to_idx.data
        copies_to_names = self.copies_to.to_name.data

        m = len(copies_to_counts)
        s = 0
        for i, name in enumerate(names):
            if i >= m:
                continue

            count = int(copies_to_counts[i])
            lst: list[tuple[int | None, str]] = []
            for _ in range(count):
                idx = int(copies_to_indices[s])
                if idx == -1:
                    idx = None

                mimic = str(copies_to_names[s])
                lst.append((idx, mimic))
                s += 1

            if lst:
                lwindow.copies_to[name] = lst

        lwindow.neighbours = tuple(
            None if idx == -1 else idx
            for idx in map(int, self.neighbours.data)
        )