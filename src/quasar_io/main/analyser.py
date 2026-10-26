__all__ = [
    "dump_analyser",
    "load_analyser"
]


from pydantic import validate_call
from quasar_errors.analysis import Analyser
from quasar_typing.literals import JSONExt
from quasar_typing.pathlib import AnyDirPath, DirPath
from quasar_utils.setup import Info

from quasar_io.file_utils import (
    Name,
    dump_to_dir_helper,
    find_file_with_name,
)

from .literals import ANALYSER_FILE_NAME

### Analyser I/O

@validate_call
def dump_analyser(
    path: AnyDirPath,
    analyser: Analyser,
    info: Info,
    name: Name = "analyser",
    json_ext: JSONExt = ".json",
    exist_ok: bool = False,
    serialization_kwargs: dict | None = None,
    compression_kwargs: dict | None = None,
) -> DirPath:
    from quasar_io.json.errors import analyser_to_file

    from .error_result import dump_error_result

    dir_path = dump_error_result(
        path,
        analyser.error_result,
        info,
        name=name,
        json_ext=json_ext,
        exist_ok=exist_ok,
        serialization_kwargs=(serialization_kwargs or {}),
        compression_kwargs=(compression_kwargs or {}),
    )
    analyser_to_file(
        dir_path / f"{ANALYSER_FILE_NAME}{json_ext}",
        analyser,
        **(serialization_kwargs or {}),
    )
    return dir_path

@validate_call
def load_analyser(
    path: DirPath,
    info: Info,
    name: Name = "analyser",
) -> Analyser:
    from quasar_io.json.errors import file_to_analyser

    from .error_result import load_error_result
    
    path = dump_to_dir_helper(path, name, exist_ok=True)
    analyser_file = find_file_with_name(path, ANALYSER_FILE_NAME)
    error_result = load_error_result(path, info, name=name)

    return file_to_analyser(analyser_file, error_result, info)