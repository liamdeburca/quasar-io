from typing import Any, Self

import pydantic
from astropy.units import Unit
from quasar_errors.analysis.measure_result import (
    FrozenMeasureResult,
    MeasureResult,
)
from quasar_typing.numpy import FloatVector


class SerializedMeasureResult(pydantic.BaseModel):
    """
    Serialized representation of a MeasureResult object.
    """    
    val: FloatVector
    unc: FloatVector
    unit: str
    frozen: bool


    @pydantic.model_validator(mode="after")
    def validate_unit(self) -> Self:
        """
        Raise a ValueError if the `.unit` is not a valid astropy Unit.
        """
        try:
            Unit(self.unit)
        except ValueError as e:
            raise ValueError(f"Invalid unit: {self.unit}") from e
        return self
        

    @pydantic.model_validator(mode="after")
    def validate_array_lengths(self) -> Self:
        """
        Raise a ValueError if the lengths of `.val` and `.unc` do not match.
        """
        n = self.val.size
        m = self.unc.size
        if n != m:
            raise ValueError(
                f"Length mismatch: `.val` has length {n}, `.unc` has length {m}"
            )
        return self


    @pydantic.field_serializer("val", "unc", when_used="json")
    def serialize_array(self, array: FloatVector) -> list[float]:
        return array.tolist()


    @classmethod
    def fromMeasureResult(
        cls, 
        measure_result: MeasureResult | FrozenMeasureResult,
    ) -> Self:
        """
        Create a SerializedMeasureResult instance from a MeasureResult object.
        """
        return cls(
            val=measure_result.val,
            unc=measure_result.unc,
            unit=str(measure_result.unit),
            frozen=isinstance(measure_result, FrozenMeasureResult),
        )


    def toMeasureResult(self) -> MeasureResult | FrozenMeasureResult:
        """
        Convert the serialized measure result back to a MeasureResult object.
        """
        cls = FrozenMeasureResult if self.frozen else MeasureResult
        return cls(
            val=self.val,
            unc=self.unc,
            unit=Unit(self.unit),
        )

    ###


    def astype(self, dtype: type) -> Self:
        """
        Return a new SerializedMeasureResult instance with `.val` and `.unc` 
        cast to the specified dtype.
        """
        return self.__class__(
            val=self.val.astype(dtype),
            unc=self.unc.astype(dtype),
            unit=self.unit,
            frozen=self.frozen,
        )


    def to(self, unit: Any) -> Self:
        """
        Return a new SerializedMeasureResult instance with `.val` and `.unc` 
        converted to the specified unit.
        """
        old_unit = Unit(self.unit)
        new_unit = Unit(unit)
        return self.__class__(
            val=(self.val * old_unit).to(new_unit).value,
            unc=(self.unc * old_unit).to(new_unit).value,
            unit=str(new_unit),
            frozen=self.frozen,
        )


class SerializedMeasureResultDict(pydantic.BaseModel):
    results: dict[str, SerializedMeasureResult]


    @classmethod
    def fromMeasureResultDict(
        cls, 
        results: dict[str, MeasureResult | FrozenMeasureResult],
    ) -> Self:
        return cls(results={
            key: SerializedMeasureResult.fromMeasureResult(measure_result)
            for key, measure_result in results.items()
        })

    def toMeasureResultDict(
        self,
    ) -> dict[str, MeasureResult | FrozenMeasureResult]:
        return {
            key: result.toMeasureResult()
            for key, result in self.results.items()
        }
