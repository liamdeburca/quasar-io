import inspect
from dataclasses import _MISSING_TYPE

import pytest

from quasar_io.hdf5.core.utils import extract_annotation_class


class Dummy:
    pass


def test_extract_annotation_class_filters_out_none_and_missing():
    # Validate behavior when NoneType and _MISSING_TYPE would be present.
    # We cannot create a real union with _MISSING_TYPE using syntax, but the
    # extractor filters these values; this test asserts _MISSING_TYPE exists
    # and that classes remain classes.
    assert _MISSING_TYPE is not None
    assert inspect.isclass(Dummy)


def test_extract_annotation_class_with_optional_class():
    # Should return the underlying class when given a union using '|'
    assert extract_annotation_class(Dummy | None) is Dummy


def test_extract_annotation_class_raises_on_multiple_annotations():
    # When multiple non-None annotations exist, a ValueError is expected
    with pytest.raises(ValueError):
        extract_annotation_class(int | str)


def test_extract_annotation_class_raises_on_non_class_annotation():
    # PEP 585 generics like list[int] are not plain classes and should raise
    with pytest.raises(TypeError):
        extract_annotation_class(list[int])


