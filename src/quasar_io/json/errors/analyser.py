from typing import Annotated, Self

import pydantic
from quasar_errors.analysis import Analyser
from quasar_errors.error_result import ErrorResult
from quasar_typing.literals import FWHMStrategy
from quasar_utils.setup import Info


class SerializedAnalyser(pydantic.BaseModel):
    """
    Serialized representation of an Analyser object.

    As the `ErrorResult` instance is saved separately (in HDF5 format), only
    specific hyperparameters of the `Analyser` are serialized.
    """
    res: Annotated[int, pydantic.Field(gt=0)]
    render_width: Annotated[float, pydantic.Field(gt=0)]
    fwhm_strategy: FWHMStrategy
    v_int: Annotated[float, pydantic.Field(ge=0)]
    ipv_int: Annotated[float, pydantic.Field(ge=0, le=100)]
    dx_int: Annotated[float, pydantic.Field(ge=0)]


    @pydantic.model_validator(mode="after")
    def validate_integration_window(self) -> Self:
        """
        Raise a ValueError if none of the integration window parameters are 
        greater than 0.
        """
        if not any([self.v_int > 0, self.ipv_int > 0, self.dx_int > 0]):
            raise ValueError(
                f"At least of of `.v_int` (`{self.v_int}`), "
                f"`.ipv_int` (`{self.ipv_int}`), or "
                f"`.dx_int` (`{self.dx_int}`) must be greater than 0."
            )
        return self


    @classmethod
    def fromAnalyser(cls, analyser: Analyser) -> Self:
        return cls(
            res=analyser.res,
            render_width=analyser.render_width,
            fwhm_strategy=analyser.fwhm_strategy,
            v_int=analyser.v_int,
            ipv_int=analyser.ipv_int,
            dx_int=analyser.dx_int,
        )


    def toAnalyser(self, error_result: ErrorResult, info: Info) -> Analyser:
        return Analyser(
            error_result,
            pool=None,
            info=info,
            **self.model_dump()
        )
