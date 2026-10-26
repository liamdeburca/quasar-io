from typing import Literal, Self, Union

import pydantic
from quasar_models.modeling import BaseModel, PrepareModel
from quasar_typing.astropy import CompoundModelType
from quasar_utils.setup import Info
from scipy.sparse import issparse

from .. import core


class PrepareModelAttrs(core.UnitHashAttrs):
    class_name: Literal["PrepareModel"] = "PrepareModel"
    is_logbinned: bool | None = None


class PrepareModelGroup(core.BaseGroup):
    attrs: PrepareModelAttrs

    x: core.FloatVectorDataset
    eval_mask: core.BooleanVectorDataset
    comp_mask: core.BooleanVectorDataset

    kernels: core.OptionalFloatMatrixDataset | None = None


    @pydantic.model_validator(mode="after")
    def validate_array_lengths(self) -> Self:
        n = self.x.data.shape[0]
        for field_name in ("eval_mask", "comp_mask", "kernels"):
            field_value: core.BaseDataset | None = getattr(self, field_name)
            if field_value is None:
                continue
            m = field_value.data.shape[0]
            if m != n:
                raise ValueError(
                    f"Field `{field_name}` has length {m}, expected {n}"
                )
        return self


    @classmethod
    def fromPrepareModel(
        cls,
        prepare_model: PrepareModel,
        info: Info,
    ) -> Self:
        attrs = PrepareModelAttrs.fromInfo(info)
        attrs.is_logbinned = prepare_model.is_logbinned
    
        kernels = prepare_model.kernels
        if issparse(kernels):
           kernels = kernels.toarray()
        
        x = core.FloatVectorDataset.fromData(prepare_model.x)
        eval_mask = core.BooleanVectorDataset.fromData(prepare_model.eval_mask)
        comp_mask = core.BooleanVectorDataset.fromData(prepare_model.comp_mask)

        if kernels is not None:
            kernels = core.FloatMatrixDataset.fromData(kernels)

        return cls(
            attrs=attrs,
            x=x,
            eval_mask=eval_mask,
            comp_mask=comp_mask,
            kernels=kernels
        )


    def toPrepareModel(
        self,
        model: Union[BaseModel, CompoundModelType[BaseModel]],
        info: Info,
    ) -> PrepareModel:
        
        self.attrs.validate_info_instance(info)
        x = self.x.data
        eval_mask = self.eval_mask.data
        comp_mask = self.comp_mask.data
        kernels = None if self.kernels is None else self.kernels.data

        return PrepareModel(
            x=x,
            model=model,
            eval_mask=eval_mask,
            comp_mask=comp_mask,
            kernels=kernels,
            info=info,
            is_logbinned=self.attrs.is_logbinned,
        )
        