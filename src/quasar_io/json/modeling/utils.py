from abc import ABC, abstractmethod
from functools import partial
from math import inf, isfinite
from typing import ClassVar, Literal, Self, final, overload

import pydantic
from astropy.modeling import Parameter
from astropy.units import Unit
from quasar_models.modeling import LinearTie
from quasar_models.modeling.template import BaseTemplate
from quasar_utils.setup import Info


@overload
def _convert(
    value: None,
    from_unit: str | None = None,
    to_unit: str | None = None,
) -> None: ...


@overload
def _convert(
    value: float, 
    from_unit: str | None = None, 
    to_unit: str | None = None,
) -> float: ...


@final
def _convert(
    value: float | None, 
    from_unit: str | None, 
    to_unit: str | None,
) -> float | None:
    if value is None or not isfinite(value):
        return None

    if from_unit and to_unit:
        value = (value * Unit(from_unit)).to(to_unit).value
    elif bool(from_unit) ^ bool(to_unit):
        raise ValueError(f"Incompatible units: from_unit={from_unit}, to_unit={to_unit}")
    
    return value


def _validate_unit(unit: str | None) -> str | None:
    if unit is None:
        return None

    unit = unit.strip()
    if not unit:
        return None
    
    try:
        Unit(unit)
    except ValueError:
        raise ValueError(f"Invalid unit: {unit}")
    return unit


def _validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Name must not be empty.")
    return name


def _validate_bounds(bounds: tuple[float | None, float | None]) -> tuple[float | None, float | None]:
    _lb = -inf if bounds[0] is None else bounds[0]
    _ub = inf if bounds[1] is None else bounds[1]
    if _ub <= _lb:
        raise ValueError(f"Bounds must be increasing: {bounds=}")
    return bounds


### Quantity


class SerializedQuantity(pydantic.BaseModel):
    value: float    
    unit: str | None = None

    scientific: ClassVar[Literal[False]] = False

    @pydantic.field_validator("unit")
    @classmethod
    def validate_unit(cls, unit: str | None) -> str | None:
        return _validate_unit(unit)

    ### Serialization
        
    @classmethod
    def __class_getitem__(
        cls, 
        mode: Literal["scientific", "plain"],
    ) -> type["SerializedQuantity | SciSerializedQuantity"]:
        return SciSerializedQuantity \
            if mode == "scientific" \
            else SerializedQuantity
    
    @pydantic.field_serializer("value")
    @classmethod
    def serialize_value(cls, value: float) -> str | float:
        return f"{value:e}" if cls.scientific else value

    ###

    def to_quantity(self, unit: str | None = None) -> float:
        return _convert(self.value, from_unit=self.unit, to_unit=unit)


class SciSerializedQuantity(SerializedQuantity):
    scientific: ClassVar[Literal[True]] = True


### Tied


class SerializedTied(pydantic.BaseModel):
    a: float
    b: float
    model_name: str
    parameter_name: str

    ### Validation

    @pydantic.field_validator("model_name", "parameter_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return _validate_name(value)

    ###

    @classmethod
    def from_tied(cls, tied: LinearTie) -> Self:
        return cls(
            a=tied.a,
            b=tied.b,
            model_name=tied.model_name,
            parameter_name=tied.parameter_name,
        )

    def to_tied(self) -> LinearTie:
        return LinearTie(
            a=self.a,
            b=self.b,
            model_name=self.model_name,
            parameter_name=self.parameter_name,
        )


class SerializedParameter(pydantic.BaseModel):
    name: str
    value: float
    
    fixed: bool = False
    bounds: tuple[float | None, float | None] = (None, None)
    tied: SerializedTied | None = None
    unit: str | None = None

    scientific: ClassVar[Literal[False]] = False

    ### Validation

    @pydantic.field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _validate_name(value)

    @pydantic.field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, 
        value: tuple[float | None, float | None],
    ) -> tuple[float | None, float | None]:
        return _validate_bounds(value)

    @pydantic.field_validator("unit")
    @classmethod
    def validate_unit(cls, unit: str | None) -> str | None:
        return _validate_unit(unit)

    ### Serialization

    @classmethod
    def __class_getitem__(
        cls, 
        mode: Literal["scientific", "plain"],
    ) -> type["SerializedParameter | SciSerializedParameter"]:
        return SciSerializedParameter \
            if mode == "scientific" \
            else SerializedParameter

    @pydantic.field_serializer("value")
    @classmethod
    def serialize_value(cls, value: float) -> float | str:
        return f"{value:e}" if cls.scientific else value

    @pydantic.field_serializer("bounds")
    @classmethod
    def serialize_bounds(
        cls, 
        bounds: tuple[float | None, float | None],
    ) -> tuple[float | str | None, float | str | None] | str:
        lb, ub = bounds
        if cls.scientific and lb is not None:
            lb = f"{lb:e}"
        if cls.scientific and ub is not None:
            ub = f"{ub:e}"
        return (lb, ub)

    ###

    @classmethod
    def from_parameter(cls, parameter: Parameter, unit: str | None = None) -> Self:
        return cls(
            name=str(parameter.name),
            value=float(parameter.value),
            fixed=bool(parameter.fixed),
            bounds=(
                None if parameter.min is None else float(parameter.min),
                None if parameter.max is None else float(parameter.max),
            ),
            tied=SerializedTied.from_tied(parameter.tied) \
                if isinstance(parameter.tied, LinearTie) \
                else None,
            unit=unit,
        )

    def to_parameter(self, parameter: Parameter, unit: str | None = None) -> None:
        """
        Inplace modification of a Parameter instance.
        """
        assert parameter.name == self.name
        convert = partial(_convert, from_unit=self.unit, to_unit=unit) 

        parameter.value = convert(self.value)
        parameter.fixed = self.fixed
        parameter.bounds = tuple(map(convert, self.bounds))
        parameter.tied = None if self.tied is None else self.tied.to_tied()


class SciSerializedParameter(SerializedParameter):
    scientific: ClassVar[Literal[True]] = True


###


class SerializedTemplate(pydantic.BaseModel, ABC):
    name: str
    path: str | None
    is_logspace: bool

    @classmethod
    def from_template(cls, template: BaseTemplate) -> Self:
        return cls(
            name=template.name,
            path=None if template.path is None else str(template.path),
            is_logspace=template.is_logspace,
        )

    def _to_template_helper(self, template: BaseTemplate, info: Info) -> BaseTemplate:
        if self.is_logspace and not template.is_logspace:
            return template.createLogspace(
                sigma_res=info.loading.sigma_res,
                keep_x=True,
            )
        return template
 
    @abstractmethod
    def to_template(self, info: Info) -> BaseTemplate:
        pass