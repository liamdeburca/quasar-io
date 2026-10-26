from typing import Self

import pydantic
from quasar_models.iron import IronModel, IronTemplate
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SerializedParameter,
    SerializedTemplate,
    _validate_name,
)


class SerializedIronTemplate(SerializedTemplate):
    @classmethod
    def from_template(cls, template: IronTemplate) -> Self:
        return super().from_template(template)

    def to_template(self, info: Info) -> IronTemplate:
        template = IronTemplate.load_from_cache(name=self.name, info=info)
        template = self._to_template_helper(template, info)

        template.fwhm_norm = info.iron.fwhm_norm
        template.normalisation = None
        template.__post_init__()
        return template


class SerializedIronModel(pydantic.BaseModel):
    name: str
    template: SerializedIronTemplate

    flux: SciSerializedParameter
    fwhm: SerializedParameter
    split: SerializedParameter
    left: SerializedParameter
    right: SerializedParameter

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("flux", "fwhm", "split", "left", "right")
    @classmethod
    def validate_params(cls, param: SerializedParameter) -> SerializedParameter:
        if param.name in {"flux", "fwhm", "split"} and not param.unit:
            raise ValueError(f"Field `{param.name}` must have a unit.")
        if param.name in {"left", "right"} and param.unit:
            raise ValueError(f"Field `{param.name}` must not have a unit.")
        return param
    
    ###

    @classmethod
    def from_model(cls, model: IronModel, info: Info) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)

        template = SerializedIronTemplate.from_template(model.template)
        flux = SciSerializedParameter.from_parameter(model.flux, unit=flux_unit)
        fwhm = SerializedParameter.from_parameter(model.fwhm, unit="km/s")
        split = SerializedParameter.from_parameter(model.split, unit=wave_unit)
        left = SerializedParameter.from_parameter(model.left)
        right = SerializedParameter.from_parameter(model.right)
        
        return cls(
            name=model.name,
            template=template,
            flux=flux,
            fwhm=fwhm,
            split=split,
            left=left,
            right=right,
        )

    def to_model(self, info: Info) -> IronModel:
        wave_unit = str(info.units.wavelength_unit)
        flux_unit = str(info.units.flux_unit)

        model = IronModel(name=self.name)
        model.template = self.template.to_template(info)
        model.scale = info.iron.scale
        model.allow_interp_fitting = info.convolution.allow_interp_fitting
        model.n_scales = info.convolution.n_scales
        
        self.flux.to_parameter(model.flux, unit=flux_unit)
        self.fwhm.to_parameter(model.fwhm, unit="km/s")
        self.split.to_parameter(model.split, unit=wave_unit)
        self.left.to_parameter(model.left)
        self.right.to_parameter(model.right)

        return model
