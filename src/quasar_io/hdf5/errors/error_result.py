from collections.abc import Iterable
from itertools import starmap
from typing import Annotated, ClassVar, Literal, Self, Union

import numpy as np
import pydantic
from quasar_errors.error_result import ErrorResult, ErrorResultBuilder
from quasar_errors.model_samples import Sample
from quasar_models.modeling import BaseModel, PrepareModel, SequentialModel
from quasar_typing.astropy import CompoundModelType
from quasar_typing.numpy import FloatVector
from quasar_utils.setup import Info

from .. import core
from ..modeling.sequential_model import (
    SequentialModelAttrs,
    SequentialModelGroup,
)


class ErrorResultModel(SequentialModelGroup):
    """
    Subclass of `SequentialModelGroup` designed for representing multiple 
    SequentialModel samples. 
    """
    _is_required: ClassVar[Literal[False]] = False
    params: core.OptionalFloatMatrixDataset | None = None


    @classmethod
    def _fromParams(
        cls,
        seq_model: SequentialModel,
        params: Iterable[FloatVector],
        info: Info,
    ) -> dict:
        return {
            "attrs": SequentialModelAttrs._fromSequentialModel(seq_model, info),
            "params": core.FloatMatrixDataset._fromData(params),
        }


    @classmethod
    def fromParams(
        cls, 
        seq_model: SequentialModel, 
        params: Iterable[FloatVector],
        info: Info,
    ) -> Self:
        return cls(**cls._fromParams(seq_model, params, info))


    @pydantic.model_validator(mode="after")
    def validate_params_shape(self) -> Self:
        if self.params is not None:
            n = self.initial_values.shape[0]
            m = self.params.shape[1]
            if m != n:
                raise ValueError(
                    f"Size of `params.shape[1]` ({m}) "
                    f"must match `initial_values.shape[0]` ({n})"
                )
        return self


###


class ErrorResultAttrs(core.UnitHashAttrs):
    class_name: Literal["ErrorResult"]
    n_samples: Annotated[int, pydantic.Field(ge=1)]


    @classmethod
    def _fromErrorResult(cls, error_result: ErrorResult, info: Info) -> dict:
        data = cls._fromInfo(info)
        data |= {
            "class_name": error_result.__class__.__name__,
            "n_samples": error_result.n_samples,
        }
        return data


    @classmethod
    def fromErrorResult(cls, error_result: ErrorResult, info: Info) -> Self:
        return cls(**cls._fromErrorResult(error_result, info))


class _CoordsDataset(core.FloatMatrixDataset):
    shape: tuple[Literal[3], int] | None = None


class ErrorResultGroup(core.BaseGroup):
    attrs: ErrorResultAttrs

    coords: _CoordsDataset
    ys: core.FloatMatrixDataset
    weights: core.FloatVectorDataset

    sequential_model: ErrorResultModel | None = None


    @pydantic.model_validator(mode="after")
    def validate_dataset_shapes(self) -> Self:
        n_pix = self.coords.shape[1]
        n_samples = self.attrs.n_samples

        if self.ys.shape[0] != n_samples:
            raise ValueError(
                f"Size of axis 0 of `ys` ({self.ys.shape[0]}) must match "
                f"`n_samples` ({n_samples})"
            )
        if self.ys.shape[1] != n_pix:
            raise ValueError(
                f"Size of axis 1 of `ys` ({self.ys.shape[1]}) must match "
                f"`n_pix` ({n_pix})"
            )
        if self.weights.shape[0] != n_samples:
            raise ValueError(
                f"Size of `weights` ({self.weights.shape[0]}) must match "
                f"`n_samples` ({n_samples})"
            )
        return self

    
    @pydantic.model_validator(mode="after")
    def validate_sequential_model(self) -> Self:
        if (self.sequential_model is not None) \
            and (self.sequential_model.params is not None):
            n_samples = self.attrs.n_samples
            n_params = self.sequential_model.params.shape[0]
            if n_params != n_samples - 1:
                raise ValueError(
                    f"Size of axis 0 of `sequential_model.params` ({n_params}) "
                    f"must match `n_samples - 1` ({n_samples - 1})"
                )
        return self


    @classmethod
    def _fromErrorResult(cls, error_result: ErrorResult, info: Info) -> dict:
        data = {
            "attrs": ErrorResultAttrs._fromErrorResult(error_result, info),
            "coords": _CoordsDataset._fromData(
                [error_result.x, error_result.dy, error_result.dx]
            ),
            "ys": core.FloatMatrixDataset._fromData(
                [s.y for s in error_result]
            ),
            "weights": core.FloatVectorDataset._fromData(
                [s.weight for s in error_result]
            ),
        }
        if error_result.seq_model is not None:
            params = []
            for i in range(1, len(error_result)):
                p = error_result[i].parameters 
                if p is None:
                    break
                params.append(p)

            if params:
                data["sequential_model"] = ErrorResultModel._fromParams(
                    error_result.seq_model,
                    params,
                    info,
                )
            else:
                data["sequential_model"] = ErrorResultModel._fromSequentialModel(
                    error_result.seq_model,
                    info,
                )

        return data

    @classmethod
    def fromErrorResult(cls, error_result: ErrorResult, info: Info) -> Self:
        return cls(**cls._fromErrorResult(error_result, info))


    def toErrorResult(
        self,
        info: Info,
        model: Union[BaseModel, CompoundModelType[BaseModel], None] = None,
    ) -> ErrorResult:
        self.attrs.validate_info_instance(info)

        coords = self.coords.data
        ys = self.ys.data
        weights = self.weights.data

        if self.sequential_model is None:
            sequential_model = None
            params = None
        else:
            if model is None:
                raise ValueError(
                    "`model` must be set to load the `SequentialModel` instance."
                )

            sequential_model = self.sequential_model\
                .toSequentialModel(model, info)
            if self.sequential_model.params is None:
                params = None
            else:
                params = self.sequential_model.params.data

        x, dy, dx = coords
        args = (x, ys[0], dy, dx)
        if sequential_model is None:
            base_sample = Sample.fromSpectrumSample(*args, weight=weights[0])
        else:
            _mask = np.ones_like(coords[0], dtype=np.bool_)
            preparer = PrepareModel(
                x=coords[0],
                model=sequential_model._model,
                eval_mask=_mask,
                comp_mask=_mask,
            )
            with preparer as (sequential_model, _):
                base_sample = Sample.fromModelSample(
                    *args, sequential_model, weight=weights[0],
                )

        error_result = ErrorResultBuilder(seq_model=sequential_model)
        error_result.simple_append(base_sample)

        if params is None:
            args_iter = zip(ys[1:], weights[1:])
            def func(_y, _w) -> Sample:
                return Sample.fromSpectrumSample(
                    x, _y, dy, dx, 
                    weight=_w,
                )
        else:
            args_iter = zip(ys[1:], weights[1:], params)
            def func(_y, _w, _p) -> Sample:
                return Sample.fromParamVector(
                    x, _y, dy, dx, 
                    _p, sequential_model, 
                    weight=_w,
                )

        error_result.simple_extend(starmap(func, args_iter))
        return error_result.finalise()
