from typing import Self

import pydantic
from quasar_models.host import HostGalaxyModel, HostGalaxyTemplate
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SciSerializedQuantity,
    SerializedParameter,
    SerializedTemplate,
    _validate_name,
)


class SerializedHostGalaxyTemplate(SerializedTemplate):
    age: SciSerializedQuantity

    @classmethod
    def from_template(cls, template: HostGalaxyTemplate) -> Self:
        return cls(
            name=template.name,
            path=None if template.path is None else str(template.path),
            is_logspace=template.is_logspace,
            age=SciSerializedQuantity(value=template.age),
        )

    def to_template(self, info: Info) -> HostGalaxyTemplate:
        template = HostGalaxyTemplate.load_from_cache(
            name=self.name, 
            age=self.age.to_quantity(),
            info=info,
        )
        template = self._to_template_helper(template, info)

        template.fwhm_norm = info.host.fwhm_norm
        template.x_norm = info.host.x_norm

        template.normalisation = None
        template.__post_init__()
        return template


class SerializedHostGalaxyModel(pydantic.BaseModel):
    name: str
    template: SerializedHostGalaxyTemplate

    flux: SciSerializedParameter
    fwhm: SerializedParameter

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("flux", "fwhm")
    @classmethod
    def validate_params(cls, param: SerializedParameter) -> SerializedParameter:
        if not param.unit:
            raise ValueError(f"Field `{param.name}` must have a unit.")
        return param
    
    ###

    @classmethod
    def from_model(cls, model: HostGalaxyModel, info: Info) -> Self:
        flux_unit = str(info.units.flux_unit)

        template = SerializedHostGalaxyTemplate.from_template(model.template)
        flux = SciSerializedParameter.from_parameter(model.flux, unit=flux_unit)
        fwhm = SerializedParameter.from_parameter(model.fwhm, unit="km/s")

        return cls(
            name=model.name,
            template=template,
            flux=flux,
            fwhm=fwhm,
        )

    def to_model(self, info: Info) -> HostGalaxyModel:
        flux_unit = str(info.units.flux_unit)

        model = HostGalaxyModel(name=self.name)
        model.template = self.template.to_template(info)
        model.allow_interp_fitting = info.convolution.allow_interp_fitting
        model.n_scales = info.convolution.n_scales
        
        self.flux.to_parameter(model.flux, unit=flux_unit)
        self.fwhm.to_parameter(model.fwhm, unit="km/s")

        return model
