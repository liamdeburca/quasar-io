import math
from itertools import chain
from typing import Any, ClassVar, Literal, Self

import numpy as np
import pydantic
from numpy.typing import ArrayLike
from pydantic_core import PydanticCustomError
from quasar_decrypt.utils._spec import _Spec

from ..core import (
    BaseAttrs,
    BaseGroup,
    BooleanVectorDataset,
    FloatVectorDataset,
    OptionalBooleanVectorDataset,
    OptionalFloatVectorDataset,
)


class XBoundsDataset(FloatVectorDataset):
    shape: tuple[Literal[2],] = (2,)

    @pydantic.model_validator(mode="after")
    def validate_bounds_are_finite_and_increasing(self) -> Self:
        if self.data is not None:
            lb, ub = self.data
            assert math.isfinite(lb), f"Lower bound must be finite: {lb=}"
            assert math.isfinite(ub), f"Upper bound must be finite: {ub=}"
            assert lb < ub, f"Bounds are not increasing: {lb=}, {ub=}"
        return self

    @classmethod
    def _fromData(cls, data: ArrayLike) -> dict:
        return super()._fromData(np.array(data, np.floating))


###


class SpecAttrs(BaseAttrs):
    x0: float
    y0: float
    has_original: bool

    @classmethod
    def _fromSpec(cls, value: _Spec) -> dict:
        return {
            "x0": value.x0,
            "y0": value.y0,
            "has_original": not value._no_original,
        }

    @classmethod
    def fromSpec(cls, value: _Spec) -> Self:
        return cls(**cls._fromSpec(value))


class BaseSpecGroup(BaseGroup):
    attrs: SpecAttrs

    # Required datasets
    x: FloatVectorDataset | None = None
    y: FloatVectorDataset | None = None
    dy: FloatVectorDataset | None = None
    dx: FloatVectorDataset | None = None

    x_log: FloatVectorDataset | None = None
    y_log: FloatVectorDataset | None = None
    dy_log: FloatVectorDataset | None = None

    # Optional datasets
    y_pl: OptionalFloatVectorDataset | None = None
    y_fe: OptionalFloatVectorDataset | None = None
    y_ba: OptionalFloatVectorDataset | None = None
    y_hg: OptionalFloatVectorDataset | None = None
    y_em: OptionalFloatVectorDataset | None = None

    kernels: OptionalFloatVectorDataset | None = None

    y_smooth: OptionalFloatVectorDataset | None = None
    p_absorbed: OptionalFloatVectorDataset | None = None

    rejected_pixels: OptionalBooleanVectorDataset | None = None
    absorbed_pixels: OptionalBooleanVectorDataset | None = None
    valid_pixels: OptionalBooleanVectorDataset | None = None
    log_valid_pixels: OptionalBooleanVectorDataset | None = None

    x_bounds: XBoundsDataset | None = None

    @pydantic.model_validator(mode="after")
    def validate_required_datasets_exist_if_x_exists(self) -> Self:
        if self.x is not None:
            required_datasets = self.get_required_dataset_fields()
            required_datasets.pop("x")

            for field_name in required_datasets:
                if getattr(self, field_name) is None:
                    raise PydanticCustomError(
                        "validation_error",
                        "Dataset `{field_name}` must exist if `x` exists",
                        {"field_name": field_name},
                    )
        return self

    @pydantic.model_validator(mode="after")
    def validate_correct_shapes(self) -> Self:
        if self.x is not None:
            required_datasets = filter(
                lambda n: n not in {"x", "x_bounds"},
                self.get_required_dataset_fields()
            )
            optional_datasets = filter(
                lambda n: getattr(self, n) is not None,
                self.get_optional_dataset_fields(),
            )
            
            n_pix: int = self.x.shape[0]
            def inner(field_name: str) -> None:
                field_shape = getattr(self, field_name).shape
                if field_shape[0] != n_pix:
                    raise PydanticCustomError(
                        "validation_error",
                        "Dataset must have the same size along axis 0 as `x`. "
                        "Shape of {field_name}: {field_shape} != ({n_pix},...)",
                        {
                            "field_name": field_name,
                            "field_shape": field_shape,
                            "n_pix": n_pix
                        }
                    )

            for field_name in chain(required_datasets, optional_datasets):
                inner(field_name)

        return self


    @classmethod
    def _from_spec_helper(cls, value: _Spec, suffix: str) -> dict:
        data = {
            "attrs": SpecAttrs._fromSpec(value),
            "x_bounds": XBoundsDataset._fromData(value.x_bounds)
        }
        arr = getattr(value, f"_x{suffix}")
        data["x"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y{suffix}")
        data["y"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_dy{suffix}")
        data["dy"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_dx{suffix}")
        data["dx"] = FloatVectorDataset._fromData(arr)

        arr = getattr(value, f"_x_log{suffix}")
        data["x_log"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y_log{suffix}")
        data["y_log"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_dy_log{suffix}")
        data["dy_log"] = FloatVectorDataset._fromData(arr)

        arr = getattr(value, f"_y_pl{suffix}")
        if (arr != 0).any():
            data["y_pl"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y_fe{suffix}")
        if (arr != 0).any():
            data["y_fe"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y_ba{suffix}")
        if (arr != 0).any():
            data["y_ba"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y_hg{suffix}")
        if (arr != 0).any():
            data["y_hg"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_y_em{suffix}")
        if (arr != 0).any():
            data["y_em"] = FloatVectorDataset._fromData(arr)
        
        arr = getattr(value, f"_kernels{suffix}")
        if arr.shape[1] > 1:
            data["kernels"] = FloatVectorDataset._fromData(arr)

        arr = getattr(value, f"_y_smooth{suffix}")
        if not np.isnan(arr).all():
            data["y_smooth"] = FloatVectorDataset._fromData(arr)
        arr = getattr(value, f"_p_absorbed{suffix}")
        if (arr != 0).any():
            data["p_absorbed"] = FloatVectorDataset._fromData(arr)

        arr = getattr(value, f"_rejected_pixels{suffix}")
        if arr.any():
            data["rejected_pixels"] = BooleanVectorDataset._fromData(arr)
        arr = getattr(value, f"_absorbed_pixels{suffix}")
        if arr.any():
            data["absorbed_pixels"] = BooleanVectorDataset._fromData(arr)
        arr = getattr(value, f"_valid_pixels{suffix}")
        if not arr.all():
            data["valid_pixels"] = BooleanVectorDataset._fromData(arr)
        arr = getattr(value, f"_log_valid_pixels{suffix}")
        if not arr.all():
            data["log_valid_pixels"] = BooleanVectorDataset._fromData(arr)

        return data

    
    def _to_spectrum_helper(self) -> dict[str, Any]:
        data = {
            "x0": float(self.attrs.x0),
            "y0": float(self.attrs.y0),
            "x_bounds": tuple(map(float, self.x_bounds.data)),
            "x": self.x.data,
            "y": self.y.data,
            "dy": self.dy.data,
            "dx": self.dx.data,

            "x_log": self.x_log.data,
            "y_log": self.y_log.data,
            "dy_log": self.dy_log.data,
        }
        if self.kernels is None:
            data["kernels"] = np.ones_like(data["x"])[:,None]
        else:
            data["kernels"] = self.kernels.data

        if self.y_smooth is not None:
            data["y_smooth"] = self.y_smooth.data
        if self.p_absorbed is not None:
            data["p_absorbed"] = self.p_absorbed.data

        if self.y_pl is not None:
            data["y_pl"] = self.y_pl.data
        if self.y_fe is not None:
            data["y_fe"] = self.y_fe.data
        if self.y_ba is not None:
            data["y_ba"] = self.y_ba.data
        if self.y_hg is not None:
            data["y_hg"] = self.y_hg.data
        if self.y_em is not None:
            data["y_em"] = self.y_em.data

        if self.rejected_pixels is not None:
            data["rejected_pixels"] = self.rejected_pixels.data
        if self.absorbed_pixels is not None:
            data["absorbed_pixels"] = self.absorbed_pixels.data
        if self.valid_pixels is not None:
            data["valid_pixels"] = self.valid_pixels.data
        if self.log_valid_pixels is not None: 
            data["log_valid_pixels"] = self.log_valid_pixels.data

        return data


class OriginalSpecGroup(BaseSpecGroup):
    _is_required: ClassVar[Literal[False]] = False


class SpecGroup(BaseSpecGroup):
    original: OriginalSpecGroup | None = None
    _is_required: ClassVar[Literal[True]] = True

    @pydantic.model_validator(mode="after")
    def validate_original_is_not_redundant(self) -> Self:
        """
        Validate that the x-arrays are different
        """
        if (self.x is not None) \
            and self.original is not None \
            and self.original.x is not None:
            my_x_arr = self.x.data
            original_x_arr = self.original.x.data
            if np.array_equal(my_x_arr, original_x_arr):
                raise ValueError(
                    "The `original.x` array must be different from the "
                    "main `x` array."
                )
        return self

    @pydantic.model_validator(mode="after")
    def validate_attrs_agrees(self) -> Self:
        if self.x is None or self.attrs is None:
            return self

        attrs_has_original = self.attrs.has_original
        if attrs_has_original and self.original is None:
            raise ValueError(
                "The `attrs.has_original` flag is True, but the `original` "
                "dataset is missing."
            )
        if not attrs_has_original and self.original is not None:
            raise ValueError(
                "The `attrs.has_original` flag is False, but the `original` "
                "dataset is present."
            )

        return self


    @classmethod
    def _fromSpec(cls, spec: _Spec) -> dict:
        data = BaseSpecGroup._from_spec_helper(spec, "")
        if not spec._no_original:
            original_data = BaseSpecGroup._from_spec_helper(spec, "_original")
            # original_data.pop("attrs")
            # original_data.pop("x_bounds")
            data["original"] = original_data
        return data


    @classmethod
    def fromSpec(cls, spec: _Spec) -> Self:
        return cls(**cls._fromSpec(spec))