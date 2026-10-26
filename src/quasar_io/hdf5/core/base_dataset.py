"""
Base classes with proper validation using pydantic.
"""
from typing import ClassVar, Literal, Self

import h5py
import numpy as np
import pydantic
from numpy.typing import ArrayLike, DTypeLike
from pydantic import Field
from pydantic_core import PydanticCustomError, PydanticUndefined
from quasar_typing.h5py import DatasetType
from quasar_typing.numpy import Array

from .base_attrs import BaseAttrs
from .utils import extract_annotation_class


class BaseDataset(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    dataset: DatasetType | None = Field(None, exclude=True, repr=False)
    attrs: BaseAttrs | None = None

    dtype: DTypeLike
    shape: tuple[int, ...]
    data: Array

    # Class variables
    _is_subdtype_of: ClassVar[DTypeLike]
    _is_required: ClassVar[Literal[True]] = True

    @pydantic.field_validator("dtype", mode="after")
    @classmethod
    def validate_dtype(cls, dtype: DTypeLike) -> DTypeLike:
        if not np.issubdtype(dtype, cls._is_subdtype_of):
            raise PydanticCustomError(
                "validation_error",
                "`dtype` ({dtype}) must be a subdtype of {_is_subdtype_of}",
                {
                    "dtype": dtype,
                    "_is_subdtype_of": cls._is_subdtype_of,
                }
            )
        return dtype

    @pydantic.field_serializer("dtype")
    def serialize_dtype(cls, dtype: DTypeLike) -> str:
        return str(dtype)

    @pydantic.field_serializer("data")
    def serialize_data(cls, data: np.ndarray) -> list:
        return data.tolist()

    ### Factory methods

    @classmethod
    def _fromData(cls, data: ArrayLike, dtype: DTypeLike | None = None) -> dict:
        data = np.array(data)
        if dtype is not None:
            data = data.astype(dtype)
        out = {
            "dataset": None,
            "attrs": None,
            "data": data,
            "dtype": data.dtype,
            "shape": data.shape,
        }
        return out

    @classmethod
    def fromData(cls, data: ArrayLike, dtype: DTypeLike | None = None) -> Self:
        return cls(**cls._fromData(data, dtype=dtype))


    @classmethod
    def _fromHDF5(cls, dataset: h5py.Dataset) -> dict:
        if np.issubdtype(cls._is_subdtype_of, np.str_):
            data = dataset.asstr()[()]
        else:
            data = dataset[()]

        default_dtype = cls.model_fields["dtype"].get_default()
        if default_dtype is PydanticUndefined:
            default_dtype = cls._is_subdtype_of

        data = data.astype(default_dtype)

        attrs_cls: type[BaseAttrs] = extract_annotation_class(
            cls.model_fields["attrs"].annotation,
        )
        return {
            "dataset": dataset,
            "attrs": attrs_cls._fromHDF5(dataset),
            "data": data,
            "dtype": data.dtype,
            "shape": data.shape,
        }


    @classmethod
    def fromHDF5(cls, dataset: h5py.Dataset) -> Self:
        return cls(**cls._fromHDF5(dataset))

    
    def toHDF5(
        self,
        group: h5py.Group | h5py.File,
        name: str,
        dtype: DTypeLike | None = None,
        **compression_kwargs,
    ) -> h5py.Dataset:
        if np.issubdtype(self._is_subdtype_of, np.str_):
            dtype = h5py.string_dtype(encoding="utf-8")
        else:
            if dtype is None:
                dtype = self.dtype
            else:
                dtype = np.dtype(dtype)
                if self._is_subdtype_of is not None \
                    and not np.issubdtype(dtype, self._is_subdtype_of):
                    raise TypeError(
                        f"`dtype` ({self.dtype}) must be a subdtype of "
                        f"{self._is_subdtype_of}"
                    )

        data = self.data.astype(dtype)

        try:
            data = self.data.astype(dtype)
            dataset = group.create_dataset(
                name,
                data=data,
                dtype=dtype,
                **compression_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to write dataset '{name}' to HDF5: {dtype=}, {data=}"
            ) from exc

        if self.attrs is not None:
            self.attrs.toHDF5(dataset)

        return dataset


class OptionalBaseDataset(BaseDataset):
    _is_required: ClassVar[Literal[False]] = False