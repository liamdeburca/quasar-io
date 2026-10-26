__all__ = [
    "dump_spectrum",
    "load_spectrum"
]

from pydantic import validate_call
from quasar_decrypt import Spectrum
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
    MODEL_FILE_NAME,
    SPECTRUM_FILE_NAME,
    SPECTRUM_GROUP_NAME,
)

### Spectrum I/O

@validate_call
def dump_spectrum(
    path: AnyDirPath,
    spectrum: Spectrum,
    name: Name = "spectrum",
    json_ext: JSONExt = ".json",
    exist_ok: bool = False,
    serialization_kwargs: dict | None = None,
    compression_kwargs: dict | None = None,
) -> DirPath:
    """
    Dump a Spectrum object to a directory containing a JSON model file and an 
    HDF5 spectrum file.

    Returns
    -------
    DirPath
        The path to the directory containing the dumped files.
    """
    from h5py import File

    from quasar_io.hdf5 import SpectrumGroup
    from quasar_io.json.modeling import model_to_file

    path = dump_to_dir_helper(path, name, exist_ok=exist_ok)
    temp_dir = AtomicDirectoryHelper(path, exist_ok=exist_ok)

    with temp_dir:
        model = spectrum.getModel()
        if model is not None:
            model_to_file(
                temp_dir.path / f"{MODEL_FILE_NAME}{json_ext}", 
                model, 
                spectrum.info, 
                **(serialization_kwargs or {}),
            )

        with File(temp_dir.path / f"{SPECTRUM_FILE_NAME}.h5", "w") as f:
            SpectrumGroup\
                .fromSpectrum(spectrum)\
                .toHDF5(f, SPECTRUM_GROUP_NAME, **(compression_kwargs or {}))

        temp_dir.commit()

    return path


@validate_call
def load_spectrum(
    path: DirPath,
    info: Info,
    name: Name = "spectrum",
) -> Spectrum:
    """
    Load a Spectrum object from a directory containing a JSON model file and an 
    HDF5 spectrum file.

    Returns
    -------
    Spectrum
        The loaded Spectrum object.
    """
    from h5py import File

    from quasar_io.hdf5 import SpectrumGroup, add_model_to_spectrum
    from quasar_io.json.modeling import file_to_model

    path = dump_to_dir_helper(path, name, exist_ok=True)
    spectrum_file = find_file_with_name(path, SPECTRUM_FILE_NAME)

    with File(spectrum_file, "r") as f:
        spectrum = SpectrumGroup\
            .fromHDF5(f[SPECTRUM_GROUP_NAME])\
            .toSpectrum(info)

    try:
        model_file = find_file_with_name(path, MODEL_FILE_NAME)
        model = file_to_model(model_file, info)
    except FileNotFoundError:
        model = None

    if model is not None:
        add_model_to_spectrum(spectrum, model)

    return spectrum