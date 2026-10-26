__all__ = [
    "BaseAttrs",
    "BaseDataset",
    "BaseGroup",
    "OptionalBaseDataset",
    "OptionalBaseGroup",
    "UnitHashAttrs",
]
from typing import ClassVar, Self

import numpy as np
import pydantic
from pydantic_core import PydanticCustomError
from quasar_utils.setup import Info

from .base_attrs import BaseAttrs
from .base_dataset import BaseDataset, OptionalBaseDataset
from .base_group import BaseGroup, OptionalBaseGroup


class UnitHashAttrs(BaseAttrs):
    hash_algo: str
    hash_value: str

    @pydantic.model_validator(mode="after")
    def validate_hash_is_set(self) -> Self:
        if getattr(self, "hash_algo", None) is None:
            raise PydanticCustomError(
                "validation_error",
                "`hash_algo` is not set."
            )
        if getattr(self, "hash_value", None) is None:
            raise PydanticCustomError(
                "validation_error",
                "`hash_value` is not set."
            )
        return self

    @classmethod
    def _fromInfo(cls, info: Info, hash_algo: str = "sha256") -> dict:
        return {
            "hash_algo": hash_algo,
            "hash_value": info.units.create_hash(hash_algo),
        }

    @classmethod
    def fromInfo(cls, info: Info, hash_algo: str = "sha256") -> Self:
        return cls(**cls._fromInfo(info, hash_algo))

    def validate_info_instance(self, info: Info) -> None:
        if self.hash_value != info.units.create_hash(self.hash_algo):
            raise ValueError(
                f"UnitsInfo hash ({self.hash_algo}) does not match the "
                f"expected hash ({self.hash_value})."
            )

### Required datasets


class BaseFloatDataset(BaseDataset):
    """
    Base class for datasets containing floating-point data.
    """
    _is_subdtype_of: ClassVar[np.floating] = np.floating


class BaseIntegerDataset(BaseDataset):
    """
    Base class for datasets containing integer data.
    """
    _is_subdtype_of: ClassVar[np.integer] = np.integer

class BaseBooleanDataset(BaseDataset):
    """
    Base class for datasets containing boolean data.
    """
    _is_subdtype_of: ClassVar[np.bool_] = np.bool_


class BaseStringDataset(BaseDataset):
    """
    Base class for datasets containing string data.
    """
    _is_subdtype_of: ClassVar[np.str_] = np.str_


### Optional datasets


class OptionalBaseFloatDataset(OptionalBaseDataset, BaseFloatDataset):
    """
    Optional base class for datasets containing floating-point data.
    """
    

class OptionalBaseIntegerDataset(OptionalBaseDataset, BaseIntegerDataset):
    """
    Optional base class for datasets containing integer data.
    """

class OptionalBaseBooleanDataset(OptionalBaseDataset, BaseBooleanDataset):
    """
    Optional base class for datasets containing boolean data.
    """

class OptionalBaseStringDataset(OptionalBaseDataset, BaseStringDataset):
    """
    Optional base class for datasets containing string data.
    """


### Subclasses


class FloatVectorDataset(BaseFloatDataset):
    shape: tuple[int] | None = None


class OptionalFloatVectorDataset(OptionalBaseFloatDataset):
    shape: tuple[int] | None = None


class FloatMatrixDataset(BaseFloatDataset):
    shape: tuple[int, int] | None = None


class OptionalFloatMatrixDataset(OptionalBaseFloatDataset):
    shape: tuple[int, int] | None = None


### Integer data


class IntegerVectorDataset(BaseIntegerDataset):
    shape: tuple[int] | None = None


class OptionalIntegerVectorDataset(OptionalBaseIntegerDataset):
    shape: tuple[int] | None = None


class IntegerMatrixDataset(BaseIntegerDataset):
    shape: tuple[int, int] | None = None


class OptionalIntegerMatrixDataset(OptionalBaseIntegerDataset):
    shape: tuple[int, int] | None = None


### Boolean data


class BooleanVectorDataset(BaseBooleanDataset):
    shape: tuple[int] | None = None


class OptionalBooleanVectorDataset(OptionalBaseBooleanDataset):
    shape: tuple[int] | None = None


class BooleanMatrixDataset(BaseBooleanDataset):
    shape: tuple[int, int] | None = None


class OptionalBooleanMatrixDataset(OptionalBaseBooleanDataset):
    shape: tuple[int, int] | None = None


### String data


class StringVectorDataset(BaseStringDataset):
    shape: tuple[int] | None = None


class OptionalStringVectorDataset(OptionalBaseStringDataset):
    shape: tuple[int] | None = None


class StringMatrixDataset(BaseStringDataset):
    shape: tuple[int, int] | None = None


class OptionalStringMatrixDataset(OptionalBaseStringDataset):
    shape: tuple[int, int] | None = None