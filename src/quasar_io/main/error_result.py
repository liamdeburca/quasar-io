__all__ = [
    "dump_error_result",
    "load_error_result"
]

from typing import Literal

from pydantic import validate_call
from quasar_errors.error_result import ErrorResult
from quasar_typing.literals import JSONExt
from quasar_typing.pathlib import AnyDirPath, DirPath
from quasar_utils.setup import Info

from quasar_io.file_utils import (
    AtomicDirectoryHelper,
    Name,
    dump_to_dir_helper,
    find_file_with_name,
)

from .literals import (
    ERROR_RESULT_FILE_NAME,
    ERROR_RESULT_GROUP_NAME,
    MODEL_FILE_NAME,
)

### ErrorResult I/O

@validate_call
def dump_error_result(
    path: AnyDirPath,
    error_result: ErrorResult,
    info: Info,
    name: Literal["error_result"] | Name = "error_result",
    exist_ok: bool = False,
    json_ext: JSONExt = ".json",
    serialization_kwargs: dict | None = None,
    compression_kwargs: dict | None = None,
) -> DirPath:
    from h5py import File

    from quasar_io.hdf5 import ErrorResultGroup
    from quasar_io.json.modeling import model_to_file
    
    path = dump_to_dir_helper(path, name, exist_ok=exist_ok)
    temp_dir = AtomicDirectoryHelper(path, exist_ok=exist_ok)

    with temp_dir:
        if error_result.seq_model is not None:
            model_to_file(
                temp_dir.path / f"{MODEL_FILE_NAME}{json_ext}", 
                error_result.seq_model._model,
                info, 
                **(serialization_kwargs or {}),
            )
        with File(temp_dir.path / f"{ERROR_RESULT_FILE_NAME}.h5", "w") as f:
            ErrorResultGroup\
                .fromErrorResult(error_result, info)\
                .toHDF5(f, ERROR_RESULT_GROUP_NAME, **(compression_kwargs or {}))
        temp_dir.commit()

    return path


@validate_call
def load_error_result(
    path: DirPath,
    info: Info,
    name: Literal["error_result"] | Name = "error_result",
) -> ErrorResult:    
    from h5py import File

    from quasar_io.hdf5 import ErrorResultGroup
    from quasar_io.json.modeling import file_to_model

    path = dump_to_dir_helper(path, name, exist_ok=True)
    error_result_file = find_file_with_name(path, ERROR_RESULT_FILE_NAME)

    try:
        model_file = find_file_with_name(path, MODEL_FILE_NAME)
        model = file_to_model(model_file, info)
    except FileNotFoundError:
        model = None


    with File(error_result_file, "r") as f:
        error_result = ErrorResultGroup\
            .fromHDF5(f[ERROR_RESULT_GROUP_NAME])\
            .toErrorResult(info, model=model)

    return error_result