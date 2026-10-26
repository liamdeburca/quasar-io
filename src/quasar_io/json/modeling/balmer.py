from typing import Self

import pydantic
from quasar_models.balmer import (
    BalmerContinuumTemplate,
    BalmerModel,
    BalmerSeriesTemplate,
)
from quasar_utils.setup import Info

from .utils import (
    SciSerializedParameter,
    SciSerializedQuantity,
    SerializedParameter,
    SerializedTemplate,
    _validate_name,
)


class SerializedBalmerContinuumTemplate(SerializedTemplate):
    temp: SciSerializedQuantity
    tau: float
    scale: float

    @classmethod
    def from_template(cls, template: BalmerContinuumTemplate, info: Info) -> Self:
        temp_unit = str(info.units.temp_unit)
        temp = SciSerializedQuantity(value=template.temp, unit=temp_unit)
        tau = template.tau
        scale = template.scale
        return cls(
            name=template.name,
            path=None if template.path is None else str(template.path),
            is_logspace=template.is_logspace,
            temp=temp,
            tau=tau,
            scale=scale,
        )

    def to_template(self, info: Info) -> BalmerContinuumTemplate:
        temp_unit = str(info.units.temp_unit)

        template = BalmerContinuumTemplate.load_from_cache(
            temp=self.temp.to_quantity(temp_unit),
            tau=self.tau,
            scale=self.scale,
            info=info,
        )
        template = self._to_template_helper(template, info)

        template.x_norm = info.balmer.edge
        template.fwhm_norm = info.balmer.fwhm_norm

        template.normalisation = None
        template.__post_init__()
        return template


class SerializedBalmerSeriesTemplate(SerializedTemplate):
    temp: SciSerializedQuantity
    dens: SciSerializedQuantity
    n_u_range: tuple[int, int]

    @pydantic.field_validator("n_u_range")
    @classmethod
    def validate_n_u_range(cls, n_u_range: tuple[int, int]) -> tuple[int, int]:
        if n_u_range[1] <= n_u_range[0]:
            raise ValueError(f"`n_u_range` must be increasing: {n_u_range=}")
        return n_u_range

    ###

    @classmethod
    def from_template(cls, template: BalmerSeriesTemplate, info: Info) -> Self:
        temp_unit = str(info.units.temp_unit)
        dens_unit = str(info.units.dens_unit)

        temp = SciSerializedQuantity(value=template.temp, unit=temp_unit)
        dens = SciSerializedQuantity(value=template.dens, unit=dens_unit)
        n_u_range = template.n_u_range
        return cls(
            name=template.name,
            path=None if template.path is None else str(template.path),
            is_logspace=template.is_logspace,
            temp=temp,
            dens=dens,
            n_u_range=n_u_range,
        )

    def to_template(self, info: Info) -> BalmerSeriesTemplate:
        temp_unit = str(info.units.temp_unit)
        dens_unit = str(info.units.dens_unit)

        template = BalmerSeriesTemplate.load_from_cache(
            name=self.name,
            temp=self.temp.to_quantity(temp_unit),
            dens=self.dens.to_quantity(dens_unit),
            n_u_range=self.n_u_range,
            info=info,
        )
        template = self._to_template_helper(template, info)

        template.x_norm = info.balmer.edge
        template.fwhm_norm = info.balmer.fwhm_norm

        template.normalisation = None
        template.__post_init__()
        return template


class SerializedBalmerModel(pydantic.BaseModel):
    name: str
    continuum_template: SerializedBalmerContinuumTemplate
    series_template: SerializedBalmerSeriesTemplate

    flux: SciSerializedParameter
    fwhm: SerializedParameter
    ratio: SerializedParameter

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        return _validate_name(name)

    @pydantic.field_validator("flux", "fwhm", "ratio")
    @classmethod
    def validate_params(cls, param: SerializedParameter) -> SerializedParameter:
        if param.name in ("flux", "fwhm") and not param.unit:
            raise ValueError(f"Field `{param.name}` must have a unit.")
        if param.name == "ratio" and param.unit:
            raise ValueError(f"Field `{param.name}` must not have a unit.")
        return param
    
    ###

    @classmethod
    def from_model(cls, model: BalmerModel, info: Info) -> Self:
        flux_unit = str(info.units.flux_unit)

        continuum_template = SerializedBalmerContinuumTemplate.from_template(
            model.continuum_template, info,
        )
        series_template = SerializedBalmerSeriesTemplate.from_template(
            model.series_template, info,
        )
        flux = SciSerializedParameter.from_parameter(model.flux, unit=flux_unit)
        fwhm = SerializedParameter.from_parameter(model.fwhm, unit="km/s")
        ratio = SerializedParameter.from_parameter(model.ratio)

        return cls(
            name=model.name,
            continuum_template=continuum_template,
            series_template=series_template,
            flux=flux,
            fwhm=fwhm,
            ratio=ratio,
        )

    def to_model(self, info: Info) -> BalmerModel:
        flux_unit = str(info.units.flux_unit)

        model = BalmerModel(name=self.name)
        model.continuum_template = self.continuum_template.to_template(info)
        model.series_template = self.series_template.to_template(info)
        
        model.edge = info.balmer.edge
        model.allow_interp_fitting = info.convolution.allow_interp_fitting
        model.n_scales = info.convolution.n_scales
        
        self.flux.to_parameter(model.flux, unit=flux_unit)
        self.fwhm.to_parameter(model.fwhm, unit="km/s")
        self.ratio.to_parameter(model.ratio)

        return model
