from typing import Self

import pydantic
from quasar_models.continuum import PowerLawModel
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SerializedParameter,
    _validate_name,
)


class SerializedPowerLawModel(pydantic.BaseModel):
    name: str
    flux: SciSerializedParameter
    alpha: SerializedParameter

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("flux", "alpha")
    @classmethod
    def validate_params(cls, param: SerializedParameter) -> SerializedParameter:
        if param.name == "flux" and not param.unit:
            raise ValueError("Field `flux` must have a unit.")
        if param.name == "alpha" and param.unit:
            raise ValueError("Field `alpha` must not have a unit.")
        return param

    ###

    @classmethod
    def from_model(cls, model: PowerLawModel, info: Info) -> Self:
        flux_unit = str(info.units.flux_unit)
        flux = SciSerializedParameter.from_parameter(model.flux, unit=flux_unit)
        alpha = SerializedParameter.from_parameter(model.alpha)
        return cls(name=model.name, flux=flux, alpha=alpha)

    def to_model(self, info: Info) -> PowerLawModel:
        model = PowerLawModel(name=self.name)
        model.x0 = info.continuum.x0
        model.y0 = info.continuum.y0

        self.flux.to_parameter(model.flux, unit=str(info.units.flux_unit))
        self.alpha.to_parameter(model.alpha)

        return model
