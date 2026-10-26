from collections.abc import Generator
from pathlib import Path

import h5py
import pytest


@pytest.fixture(scope="function")
def hdf5_file(tmp_path: Path) -> Generator[h5py.File, None, None]:
    """Create a temporary HDF5 file and yield an open h5py.File object.

    The file is created in the pytest-provided temporary directory and is
    closed and removed by pytest when the fixture teardown runs.
    """
    path = tmp_path / "test.h5"
    f = h5py.File(path, "w")
    try:
        yield f
    finally:
        f.close()
        path.unlink()  # Remove the temporary file after closing
