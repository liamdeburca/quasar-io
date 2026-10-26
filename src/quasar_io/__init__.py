import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure: dict[str, list[str]] = {
    "main.spectrum": [
        "dump_spectrum", 
        "load_spectrum",
    ],
    "main.analyser": [
        "dump_analyser", 
        "load_analyser",
    ],
    "main.error_result": [
        "dump_error_result", 
        "load_error_result",
    ],
    "main.measure_result": [
        "dump_measure_result",
        "load_measure_result",
        "dump_measure_result_dict",
        "load_measure_result_dict",
    ]
}

if TYPE_CHECKING:
    from .main.analyser import dump_analyser, load_analyser  # noqa: F401
    from .main.error_result import (  # noqa: F401
        dump_error_result,
        load_error_result,
    )
    from .main.measure_result import (  # noqa: F401
        dump_measure_result,
        dump_measure_result_dict,
        load_measure_result,
        load_measure_result_dict,
    )
    from .main.spectrum import dump_spectrum, load_spectrum  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        __file__,
        _import_structure,
    )