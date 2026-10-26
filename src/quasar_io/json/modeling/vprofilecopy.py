from typing import Literal, Self

import pydantic
from quasar_models.line import VProfileCopy, VProfileCopyDict
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SerializedParameter,
    SerializedQuantity,
    _validate_name,
)


class SerializedVProfileCopyModel(pydantic.BaseModel):
    name: str
    master_name: str
    wave: SerializedQuantity
    linetype: Literal["n", "b"]
    n_sigmas: float
    n_profiles: int

    strength_scale: SerializedParameter
    strengths: list[SciSerializedParameter]
    fwhm_vs: list[SerializedParameter]
    v_offs: list[SerializedParameter]

    ### Validation

    @pydantic.field_validator("name", "master_name")
    @classmethod
    def validate_names(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("wave")
    @classmethod
    def validate_wave(cls, wave: SerializedQuantity) -> SerializedQuantity:
        if not wave.unit:
            raise ValueError("Field `wave` must have a unit.")
        return wave

    @pydantic.field_validator("strength_scale")
    @classmethod
    def validate_strength_scale(
        cls, 
        strength_scale: SerializedParameter,
    ) -> SerializedParameter:
        if strength_scale.unit:
            raise ValueError("Field `strength_scale` should not have a unit.")
        return strength_scale

    @pydantic.field_validator("strengths", "fwhm_vs", "v_offs")
    @classmethod
    def validate_param_list(
        cls,
        param_list: list[SerializedParameter],
    ) -> list[SerializedParameter]:
        if not all(p.unit for p in param_list):
            raise ValueError("All elements of the parameter list must have a unit.")
        return param_list
    
    ###

    @classmethod
    def from_model(cls, model: VProfileCopy, info: Info) -> Self:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        wave = SerializedQuantity(value=model.wave, unit=wave_unit)
        strength_scale = SerializedParameter.from_parameter(model.strength_scale)

        strengths = []
        fwhm_vs = []
        v_offs = []
        for i in range(1, model.n_profiles + 1):
            strengths.append(SciSerializedParameter.from_parameter(
                getattr(model, f"strength_{i}"), 
                unit=strength_unit,
            ))
            fwhm_vs.append(SerializedParameter.from_parameter(
                getattr(model, f"fwhm_v_{i}"), 
                unit="km/s",
            ))
            v_offs.append(SerializedParameter.from_parameter(
                getattr(model, f"v_off_{i}"), 
                unit="km/s",
            ))

        return cls(
            name=model.name,
            master_name=model.master_name,
            wave=wave,
            linetype=model.linetype,
            n_sigmas=model.n_sigmas,
            n_profiles=model.n_profiles,
            strength_scale=strength_scale,
            strengths=strengths,
            fwhm_vs=fwhm_vs,
            v_offs=v_offs,
        )


    def to_model(self, info: Info) -> VProfileCopy:
        wave_unit = str(info.units.wavelength_unit)
        strength_unit = str(info.units.strength_unit)

        cls: type[VProfileCopy] = VProfileCopyDict[self.n_profiles]
        model = cls(name=self.name)
        model.master_name = self.master_name
        model.wave = self.wave.to_quantity(unit=wave_unit)
        model.sigma_res = info.loading.sigma_res
        model.linetype = self.linetype
        model.n_sigmas = self.n_sigmas

        self.strength_scale.to_parameter(model.strength_scale)
        for i in range(1, self.n_profiles+1):
            self.strengths[i-1].to_parameter(
                getattr(model, f"strength_{i}"), 
                unit=strength_unit,
            )
            self.fwhm_vs[i-1].to_parameter(
                getattr(model, f"fwhm_v_{i}"), 
                unit="km/s",
            )
            self.v_offs[i-1].to_parameter(
                getattr(model, f"v_off_{i}"), 
                unit="km/s",
            )

        return model
