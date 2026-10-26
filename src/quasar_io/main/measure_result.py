__all__ = [
    "dump_measure_result",
    "dump_measure_result_dict",
    "load_measure_result",
    "load_measure_result_dict",
]

from typing import Literal

from pydantic import validate_call
from quasar_errors.analysis.measure_result import (
    FrozenMeasureResult,
    MeasureResult,
)
from quasar_typing.literals import JSONExt
from quasar_typing.pathlib import AnyDirPath, DirPath

from quasar_io.file_utils import (
    AtomicDirectoryHelper,
    Name,
    dump_to_dir_helper,
    find_file_with_name,
)

MEASURE_RESULT_FILE_NAME: Literal["measure_result"] = "measure_result"
MEASURE_RESULT_DICT_FILE_NAME: Literal["measure_result_dict"] = "measure_result_dict"

### (Frozen)MeasureResult I/O

@validate_call
def dump_measure_result(
    path: AnyDirPath,
    measure_result: MeasureResult | FrozenMeasureResult,
    name: Name = "measure_result",
    json_ext: JSONExt = ".json",
    exist_ok: bool = False,
    serialization_kwargs: dict | None = None,
) -> DirPath:
    """
    Dump a (Frozen)MeasureResult to a file.
    """
    from quasar_io.json.errors import measure_result_to_file

    path = dump_to_dir_helper(path, name, exist_ok=exist_ok)
    temp_dir = AtomicDirectoryHelper(path, exist_ok=exist_ok)

    with temp_dir:
        measure_result_to_file(
            temp_dir.path / f"{MEASURE_RESULT_FILE_NAME}{json_ext}",
            measure_result,
            **(serialization_kwargs or {}),
        )
        temp_dir.commit()

    return path


@validate_call
def load_measure_result(
    path: DirPath,
    name: Name = "measure_result",
) -> MeasureResult | FrozenMeasureResult:
    """
    Load a (Frozen)MeasureResult from a file.
    """
    from quasar_io.json.errors import file_to_measure_result

    path = dump_to_dir_helper(path, name, exist_ok=True)
    measure_result_file = find_file_with_name(path, MEASURE_RESULT_FILE_NAME)
    
    return file_to_measure_result(measure_result_file)


### MeasureResultDict I/O

@validate_call
def dump_measure_result_dict(
    path: AnyDirPath,
    measure_result_dict: dict[str, MeasureResult | FrozenMeasureResult],
    name: Name = "measure_result_dict",
    json_ext: JSONExt = ".json",
    exist_ok: bool = False,
    serialization_kwargs: dict | None = None,
) -> DirPath:
    """
    Dump a dictionary of (Frozen)MeasureResult objects to a file.
    """
    from quasar_io.json.errors import measure_result_dict_to_file

    path = dump_to_dir_helper(path, name, exist_ok=exist_ok)
    temp_dir = AtomicDirectoryHelper(path, exist_ok=exist_ok)

    with temp_dir:
        measure_result_dict_to_file(
            temp_dir.path / f"{MEASURE_RESULT_DICT_FILE_NAME}{json_ext}",
            measure_result_dict,
            **(serialization_kwargs or {}),
        )
        temp_dir.commit()

    return path


@validate_call
def load_measure_result_dict(
    path: DirPath,
    name: Name = "measure_result_dict",
) -> dict[str, MeasureResult | FrozenMeasureResult]:
    """
    Load a dictionary of (Frozen)MeasureResult objects from a file.
    """
    from quasar_io.json.errors import file_to_measure_result_dict
    
    path = dump_to_dir_helper(path, name, exist_ok=True)
    measure_result_dict_file = find_file_with_name(
        path, 
        MEASURE_RESULT_DICT_FILE_NAME,
    )
    
    return file_to_measure_result_dict(measure_result_dict_file)