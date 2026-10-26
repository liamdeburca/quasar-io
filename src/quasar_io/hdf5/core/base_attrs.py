from typing import Any, Self

import pydantic
from pydantic import Field
from quasar_typing.h5py import (
    AttributeManagerType,
    DatasetType,
    FileType,
    GroupType,
)


class BaseAttrs(pydantic.BaseModel):
    attrs: AttributeManagerType | None = Field(None, exclude=True, repr=False)

    @classmethod
    def _fromHDF5(
        cls, 
        hdf5_obj: DatasetType | GroupType | FileType,
    ) -> dict[str, Any]:
        data = {"attrs": hdf5_obj.attrs}

        for field_name, field_info in cls.model_fields.items():
            if field_name == "attrs":
                continue

            if field_info.is_required():
                if field_name not in hdf5_obj.attrs:
                    raise ValueError(
                        "Missing required HDF5 attribute for field "
                        f"'{field_name}'"
                    )
                data[field_name] = hdf5_obj.attrs[field_name]
            else:
                data[field_name] = hdf5_obj.attrs.get(
                    field_name,
                    field_info.get_default()
                ) 

        return data

    @classmethod
    def fromHDF5(cls, hdf5_obj: DatasetType | GroupType | FileType) -> Self:
        return cls(**cls._fromHDF5(hdf5_obj))
    
    
    def toHDF5(
        self, 
        hdf5_obj: AttributeManagerType | DatasetType | GroupType | FileType,
    ) -> AttributeManagerType:
        """
        Writes this model's fields to the HDF5 object's attributes.
        """
        from h5py import AttributeManager

        if isinstance(hdf5_obj, AttributeManager):
            attrs = hdf5_obj
        else:
            attrs = hdf5_obj.attrs

        attrs.clear()
        attrs.update(self.model_dump(exclude={"attrs"}, exclude_none=True))
        return attrs