from typing import Literal, Self, Union

from quasar_models.modeling import BaseModel, SequentialModel
from quasar_typing.astropy import CompoundModelType
from quasar_utils.setup import Info
from scipy.sparse import issparse

from .. import core


class SequentialModelAttrs(core.UnitHashAttrs):
    class_name: Literal["SequentialModel"] = "SequentialModel"

    @classmethod
    def _fromSequentialModel(
        cls, 
        seq_model: SequentialModel, 
        info: Info,
    ) -> dict:
        data = cls._fromInfo(info)
        data["class_name"] = seq_model.__class__.__name__
        return data

    @classmethod
    def fromSequentialModel(
        cls, 
        seq_model: SequentialModel, 
        info: Info,
    ) -> Self:
        return cls(**cls._fromSequentialModel(seq_model, info))


class SequentialModelGroup(core.BaseGroup): 
    attrs: SequentialModelAttrs

    names: core.StringVectorDataset
    initial_values: core.FloatVectorDataset
    kernels: core.OptionalFloatMatrixDataset | None = None


    @classmethod
    def _fromSequentialModel(
        cls,
        seq_model: SequentialModel,
        info: Info,
    ) -> dict:
        data = {
            "attrs": SequentialModelAttrs._fromSequentialModel(seq_model, info),
            "names": core.StringVectorDataset._fromData(
                [m.name for m in seq_model.submodels]
            ),
            "initial_values": core.FloatVectorDataset._fromData(
                seq_model._initial_values,
            ),
        }
        if seq_model.kernels is not None:
            kernels = seq_model.kernels
            if issparse(kernels):
                kernels = kernels.toarray()
            data["kernels"] = core.FloatMatrixDataset._fromData(kernels)

        return data


    @classmethod
    def fromSequentialModel(
        cls,
        seq_model: SequentialModel,
        info: Info,
    ) -> Self:
        return cls(**cls._fromSequentialModel(seq_model, info))


    def toSequentialModel(
        self,
        model: Union[BaseModel, CompoundModelType[BaseModel]],
        info: Info,
    ) -> SequentialModel:
        self.attrs.validate_info_instance(info)
        names = [str(n) for n in self.names.data]
        initial_values = self.initial_values.data
        kernels = None if self.kernels is None else self.kernels.data

        submodels = [model[name].copy() for name in names]
        _model = sum(submodels[1:], start=submodels[0])
        _model.parameters = initial_values
        
        return SequentialModel(_model, kernels=kernels)