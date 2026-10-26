"""
Base classes with proper validation using pydantic.
"""
from typing import ClassVar, Literal, Self

import h5py
import pydantic
from pydantic import Field
from pydantic.fields import FieldInfo
from quasar_typing.h5py import FileType, GroupType

from .base_attrs import BaseAttrs
from .base_dataset import BaseDataset
from .utils import extract_annotation_class


class BaseGroup(pydantic.BaseModel):
    group: GroupType | FileType | None = Field(None, exclude=True)
    attrs: BaseAttrs | None = None

    _is_required: ClassVar[Literal[True]] = True

    ### Datasets

    @classmethod
    def get_dataset_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.model_fields.items()
            if field_name != "group" and issubclass(
                extract_annotation_class(field_info.annotation), 
                BaseDataset,
            )
        }

    @classmethod
    def get_required_dataset_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.get_dataset_fields().items()
            if extract_annotation_class(field_info.annotation)._is_required
        }

    @classmethod
    def get_optional_dataset_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.get_dataset_fields().items()
            if not extract_annotation_class(field_info.annotation)._is_required
        }

    ### Groups

    @classmethod
    def get_group_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.model_fields.items()
            if field_name != "group" and issubclass(
                extract_annotation_class(field_info.annotation), 
                BaseGroup,
            )
        }

    @classmethod
    def get_required_group_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.get_group_fields().items()
            if extract_annotation_class(field_info.annotation)._is_required
        }

    @classmethod
    def get_optional_group_fields(cls) -> dict[str, FieldInfo]:
        return {
            field_name: field_info
            for field_name, field_info in cls.get_group_fields().items()
            if not extract_annotation_class(field_info.annotation)._is_required
        }


    ### Factory methods


    @classmethod
    def _fromHDF5(cls, group: h5py.Group | h5py.File) -> dict: 
        attrs_cls: type[BaseAttrs] = extract_annotation_class(
            cls.model_fields["attrs"].annotation,
        )
        data = {"attrs": attrs_cls._fromHDF5(group)}

        # Populate required datasets
        for name, info in cls.get_required_dataset_fields().items():
            if name not in group:
                raise ValueError(
                    f"Required dataset '{name}' is missing in the "
                    f"group/file @ {group.name}"
                )

            dataset_cls: type[BaseDataset] = extract_annotation_class(
                info.annotation,
            )
            data[name] = dataset_cls._fromHDF5(group[name])

        # Populate optional datasets
        for name, info in cls.get_optional_dataset_fields().items():
            if name not in group:
                continue

            dataset_cls: type[BaseDataset] = extract_annotation_class(
                info.annotation,
            )
            data[name] = dataset_cls._fromHDF5(group[name])

        # Populate required groups
        for name, info in cls.get_required_group_fields().items():
            if name not in group:
                raise ValueError(
                    f"Required group '{name}' is missing in the group/file "
                    f"@ {group.name}"
                )

            group_cls: type[BaseGroup] = extract_annotation_class(
                info.annotation,
            )
            data[name] = group_cls._fromHDF5(group[name])

        # Populate optional groups
        for name, info in cls.get_optional_group_fields().items():
            if name not in group:
                continue

            group_cls: type[BaseGroup] = extract_annotation_class(
                info.annotation,
            )
            data[name] = group_cls._fromHDF5(group[name])

        return data


    @classmethod
    def fromHDF5(cls, group: h5py.Group | h5py.File) -> Self:
        return cls(**cls._fromHDF5(group))


    def toHDF5(
        self,
        group: h5py.Group | h5py.File,
        name: str,
        **compression_kwargs,
    ) -> h5py.Group:

        sub = group.create_group(name)
        if self.attrs is not None:
            self.attrs.toHDF5(sub)

        # Required subgroups
        for field_name in self.get_required_group_fields():
            field_value: BaseGroup = getattr(self, field_name)
            try:
                field_value.toHDF5(sub, field_name, **compression_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to write required subgroup '{field_name}' to HDF5"
                ) from exc

        # Optional subgroups
        for field_name in self.get_optional_group_fields():
            field_value: BaseGroup | None = getattr(self, field_name)
            if field_value is None:
                continue

            try:
                field_value.toHDF5(sub, field_name, **compression_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to write optional subgroup '{field_name}' to HDF5"
                ) from exc

        # Required datasets
        for field_name in self.get_required_dataset_fields():
            field_value = getattr(self, field_name)

            try:
                field_value.toHDF5(sub, field_name, **compression_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to write required dataset '{field_name}' to HDF5"
                ) from exc

        # Optional datasets
        for field_name in self.get_optional_dataset_fields():
            field_value = getattr(self, field_name)
            if field_value is None:
                continue
            
            try:
                field_value.toHDF5(sub, field_name, **compression_kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to write optional dataset '{field_name}' to HDF5"
                ) from exc

        return sub


class OptionalBaseGroup(BaseGroup):
    _is_required: ClassVar[Literal[False]] = False
