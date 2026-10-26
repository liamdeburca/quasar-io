from pathlib import Path

from pydantic import validate_call
from quasar_errors.analysis import Analyser
from quasar_errors.analysis.measure_result import (
    FrozenMeasureResult,
    MeasureResult,
)
from quasar_errors.error_result import ErrorResult
from quasar_typing.pathlib import AnyFilePath, FilePath
from quasar_utils.setup import Info

from ..pkg_utils import load_pkg
from .analyser import SerializedAnalyser
from .measure_result import (
    SerializedMeasureResult,
    SerializedMeasureResultDict,
)

### Measure Result // Measure Result Dict


@validate_call
def measure_result_to_file(
    path: AnyFilePath,
    measure_result: MeasureResult | FrozenMeasureResult,
    **kwargs,
) -> FilePath:
    """
    Write a MeasureResult object to a file.
    """
    path = Path(path)

    pkg_name, pkg = load_pkg(path.suffix)
    if pkg_name == "toml":
        kwargs.clear()
    else:
        kwargs = {"indent": 2} | kwargs

    serialized_model = SerializedMeasureResult\
        .fromMeasureResult(measure_result)\
        .model_dump(mode="json")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        pkg.dump(serialized_model, f, **kwargs)

    return path


@validate_call
def measure_result_dict_to_file(
    path: AnyFilePath,
    measure_result_dict: dict[str, MeasureResult | FrozenMeasureResult],
    **kwargs,
) -> FilePath:
    """
    Write a dictionary of MeasureResult objects to a file.
    """
    path = Path(path)

    pkg_name, pkg = load_pkg(path.suffix)
    if pkg_name == "toml":
        kwargs.clear()
    else:
        kwargs = {"indent": 2} | kwargs
    
    serialized_model = SerializedMeasureResultDict\
        .fromMeasureResultDict(measure_result_dict)\
        .model_dump(mode="json")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        pkg.dump(serialized_model, f, **kwargs)

    return path


@validate_call
def file_to_measure_result(path: FilePath) -> MeasureResult | FrozenMeasureResult:
    """
    Read a MeasureResult object from a file.
    """
    pkg_name, pkg = load_pkg(path.suffix)
    
    if pkg_name == "yaml":
        from yaml import Loader
        args = (Loader,)
    else:
        args = ()

    with open(path, "r") as f:
        data = pkg.load(f, *args)

    return SerializedMeasureResult(**data).toMeasureResult()


@validate_call
def file_to_measure_result_dict(path: FilePath) -> dict[str, MeasureResult | FrozenMeasureResult]:
    """
    Read a dictionary of MeasureResult objects from a file.
    """
    pkg_name, pkg = load_pkg(path.suffix)
    
    if pkg_name == "yaml":
        from yaml import Loader
        args = (Loader,)
    else:
        args = ()

    with open(path, "r") as f:
        data = pkg.load(f, *args)

    return SerializedMeasureResultDict(**data).toMeasureResultDict()


### Analyser


@validate_call
def analyser_to_file(
    path: AnyFilePath,
    analyser: Analyser,
    **kwargs,
) -> FilePath:
    """
    Write an Analyser object to a file.
    """
    path = Path(path)

    pkg_name, pkg = load_pkg(path.suffix)
    if pkg_name == "toml":
        kwargs.clear()
    else:
        kwargs = {"indent": 2} | kwargs

    serialized_model = SerializedAnalyser\
        .fromAnalyser(analyser)\
        .model_dump(mode="json")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        pkg.dump(serialized_model, f, **kwargs)

    return path


@validate_call
def file_to_analyser(
    path: FilePath, 
    error_result: ErrorResult,
    info: Info,
) -> Analyser:
    """
    Read an Analyser object from a file.
    """
    pkg_name, pkg = load_pkg(path.suffix)
    
    if pkg_name == "yaml":
        from yaml import Loader
        args = (Loader,)
    else:
        args = ()

    with open(path, "r") as f:
        data = pkg.load(f, *args)

    return SerializedAnalyser(**data).toAnalyser(error_result, info)