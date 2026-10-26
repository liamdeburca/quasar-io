from typing import Literal, Self

import pydantic
from quasar_models.line import GaussianModel
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SerializedParameter,
    SerializedQuantity,
    _validate_name,
)


class SerializedGaussianModel(pydantic.BaseModel):
    name: str
    wave: SerializedQuantity
    linetype: Literal["n", "b"]
    n_sigmas: float

    strength: SciSerializedParameter
    fwhm_v: SerializedParameter
    v_off: SerializedParameter

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("wave")
    @classmethod
    def validate_wave(cls, wave: SerializedQuantity) -> SerializedQuantity:
        if not wave.unit:
            raise ValueError("Field `wave` must have a unit.")
        return wave

    @pydantic.field_validator("strength", "fwhm_v", "v_off")
    @classmethod
    def validate_params(cls, param: SerializedParameter) -> SerializedParameter:
        if not param.unit:
            raise ValueError(f"Field `{param.name}` must have a unit.")
        return param
    
    ###

    @classmethod
    def from_model(cls, model: GaussianModel, info: Info) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        wave = SerializedQuantity(value=model.wave, unit=wave_unit)
        strength = SciSerializedParameter.from_parameter(model.strength, unit=strength_unit)
        fwhm_v = SerializedParameter.from_parameter(model.fwhm_v, unit="km/s")
        v_off = SerializedParameter.from_parameter(model.v_off, unit="km/s")

        return cls(
            name=model.name,
            wave=wave,
            linetype=model.linetype,
            n_sigmas=model.n_sigmas,
            strength=strength,
            fwhm_v=fwhm_v,
            v_off=v_off,
        )

    def to_model(self, info: Info) -> GaussianModel:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        model = GaussianModel(name=self.name)
        model.wave = self.wave.to_quantity(unit=wave_unit)
        model.sigma_res = info.loading.sigma_res
        model.linetype = self.linetype
        model.n_sigmas = self.n_sigmas

        self.strength.to_parameter(model.strength, unit=strength_unit)
        self.fwhm_v.to_parameter(model.fwhm_v, unit="km/s")
        self.v_off.to_parameter(model.v_off, unit="km/s")

        return model
