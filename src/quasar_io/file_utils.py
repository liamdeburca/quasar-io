__all__ = [
    "AtomicDirectoryHelper",
    "AtomicFileHelper",
    "dump_to_dir_helper",
    "find_file_with_name",
    "get_path_helper"
]

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import StringConstraints, validate_call
from quasar_typing.pathlib import (
    AnyDirPath,
    AnyFilePath,
    DirPath,
    FilePath,
)

type Name = Annotated[str, StringConstraints(min_length=1)]


@validate_call
def get_path_helper(
    path: AnyDirPath, 
    name: Name,
) -> AnyDirPath:
    """
    Append the specified name to the given directory path if it is not already 
    the last component.

    Notes
    -----
    Pydantic validated function.
    """
    return path if path.name == name else (path / name)


@validate_call
def dump_to_dir_helper(
    path: AnyDirPath, 
    name: Name, 
    exist_ok: bool = False,
) -> DirPath:
    """
    Create a directory at the specified path with the given name.

    Notes
    -----
    Pydantic validated function.
    """
    path = get_path_helper(path, name)
    path.mkdir(parents=True, exist_ok=exist_ok)
    return path


@validate_call
def find_file_with_name(
    path: DirPath,
    name: Name,
) -> FilePath:
    """
    Find the first file with the specified name in the given directory. 

    Notes
    -----
    Pydantic validated function.
    """
    try:
        return next(path.glob(f"{name}.*"))
    except StopIteration:
        raise FileNotFoundError(
            f"No file with name '{name}' found in directory '{path}'"
        )


class AtomicFileHelper:
    """
    Context manager for creating and committing changes to a temporary file. 

    If `.commit` is called, the destination file is removed and replaced with 
    the temporary file.

    Notes
    -----
    If `.commit` is not called, this context manager is equivalent to using 
    `tempfile.NamedTemporaryFile`.
    """
    def __init__(self, dst: AnyFilePath, exist_ok: bool = False):
        self.dst = dst
        self.src: tempfile._TemporaryFileWrapper | None = None

        if self.dst.exists() and not exist_ok:
            raise FileExistsError(
                f"Destination file '{self.dst}' already exists."
            )

    def __enter__(self):
        self.src = tempfile.NamedTemporaryFile(suffix=self.dst.suffix)
        return self.src.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.src.__exit__(exc_type, exc_value, traceback)

    def commit(self):
        _ = self.path
        if self.dst.exists():
            # Remove the old file
            self.dst.unlink()

        # Replace with the new file (keep original copy for __exit__ method)
        shutil.copy(self.src, self.dst)

    @property
    def path(self) -> FilePath:
        if self.src is None:
            raise RuntimeError("Temporary file (src) does not exist.")
        return Path(self.src.name)


class AtomicDirectoryHelper:
    """
    Context manager for creating and committing changes to a temporary 
    directory.

    If `.commit` is called, the destination directory is removed and replaced 
    with the temporary directory.

    Notes
    -----
    If `.commit` is not called, this context manager is equivalent to using 
    `tempfile.TemporaryDirectory`.
    """
    def __init__(self, dst: AnyDirPath, exist_ok: bool = False):
        self.dst = dst
        self.src = None

        if self.dst.exists() and not exist_ok:
            raise FileExistsError(
                f"Destination directory '{self.dst}' already exists."
            )

    def __enter__(self):
        self.src = tempfile.TemporaryDirectory()
        return self.src.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.src.__exit__(exc_type, exc_value, traceback)

    def commit(self):
        _ = self.path
        if self.dst.exists():
            # Remove the old directory
            shutil.rmtree(self.dst)

        # Replace with the new directory (keep original copy for __exit__ method)
        shutil.copytree(self.src.name, self.dst)

    @property
    def path(self) -> DirPath:
        if self.src is None:
            raise RuntimeError("Temporary directory (src) does not exist.")
        return Path(self.src.name)