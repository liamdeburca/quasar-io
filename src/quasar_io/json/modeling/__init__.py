from typing import Union

import pydantic
from quasar_models.modeling import BaseModel
from quasar_typing.astropy import CompoundModelType
from quasar_typing.pathlib import (
    AnyJSONPath,
    AnyTOMLPath,
    AnyTOONPath,
    AnyYAMLPath,
    JSONPath,
    TOMLPath,
    TOONPath,
    YAMLPath,
)
from quasar_utils.setup import Info

from ..pkg_utils import load_pkg
from .compound_model import SerializedCompoundModel

type AnyModelPath = AnyJSONPath | AnyYAMLPath | AnyTOMLPath | AnyTOONPath
type ModelPath = JSONPath | YAMLPath | TOMLPath | TOONPath


@pydantic.validate_call
def model_to_file(
    path: AnyModelPath,
    model: Union[BaseModel, CompoundModelType[BaseModel]],
    info: Info,
    **kwargs,
) -> ModelPath:
    pkg_name, pkg = load_pkg(path.suffix)
    if pkg_name == "toml":
        kwargs.clear()
    else:
        kwargs = {"indent": 2} | kwargs

    serialized_model = SerializedCompoundModel\
        .from_compound_model(model, info) \
        .model_dump(mode="python")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        pkg.dump(serialized_model, f, **kwargs)

    return path


@pydantic.validate_call
def file_to_model(
    path: ModelPath,
    info: Info,
) -> Union[BaseModel, CompoundModelType[BaseModel]]:
    pkg_name, pkg = load_pkg(path.suffix)
    
    if pkg_name == "yaml":
        from yaml import Loader
        args = (Loader,)
    else:
        args = ()

    with open(path, "r") as f:
        serialized_model = pkg.load(f, *args)

    return SerializedCompoundModel\
        .model_validate(serialized_model)\
        .to_model(info)