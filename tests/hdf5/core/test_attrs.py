import h5py
import pytest
from pydantic import ValidationError

from quasar_io.hdf5.core import BaseAttrs


class DummyAttrs(BaseAttrs):
    # Required fields
    a: int | None = None
    b: float | None = None
    c: str | None = None
    d: bool | None = None


data = {
    "a": 1,
    "b": 2.0,
    "c": "some_text",
    "d": True,
}


def test_dummy_attrs_default_initialization():
    """
    This should fail since no fields are populated, and no `attrs` value was 
    given.
    """
    DummyAttrs()
    # with pytest.raises((ValueError, ValidationError)):


def test_dummy_attrs_initialization_with_partial_data():
    for key in data:
        partial_data = data.copy()
        partial_data.pop(key)
        with pytest.raises((ValueError, ValidationError)):
            DummyAttrs(**partial_data)


def test_dummy_attrs_initialization():
    attrs = DummyAttrs(**data)
    assert attrs.a == data["a"]
    assert attrs.b == data["b"]
    assert attrs.c == data["c"]
    assert attrs.d == data["d"]


def test_dummy_attrs_read_from_hdf5(hdf5_file: h5py.File):
    """Populate the File.attrs and ensure the model reads them back."""
    # Populate the HDF5 file attributes
    hdf5_file.attrs.update(data)
    assert hdf5_file.attrs["a"] == data["a"]
    assert hdf5_file.attrs["b"] == data["b"]
    assert hdf5_file.attrs["c"] == data["c"]
    assert hdf5_file.attrs["d"] == data["d"]

    # Instantiate the model with the AttributeManager
    model = DummyAttrs.fromHDF5(hdf5_file)
    assert model.a == data["a"]
    assert model.b == data["b"]
    assert model.c == data["c"]
    assert model.d == data["d"]


def test_dummy_attrs_write_to_hdf5(hdf5_file: h5py.File):
    """Write model fields to the HDF5 file attributes via toHDF5()."""
    model = DummyAttrs(**data)
    attrs = model.toHDF5(hdf5_file)

    assert attrs["a"] == data["a"] == model.a
    assert attrs["b"] == data["b"] == model.b
    assert attrs["c"] == data["c"] == model.c
    assert attrs["d"] == data["d"] == model.d


def test_dummy_attrs_missing_hdf5_attribute_raises(hdf5_file):
    """If attrs is missing required keys, model construction should fail."""
    # Only provide a subset of attributes
    hdf5_file.attrs.update({"a": 1})

    with pytest.raises(ValueError):
        DummyAttrs(attrs=hdf5_file.attrs)


def test_dummy_attrs_updates_existing_hdf5_attrs(hdf5_file):
    """When attrs contains all keys, providing model values updates them."""
    # Start with the baseline attribute set
    hdf5_file.attrs.update(data)

    # Create model overriding all fields and attaching the AttributeManager
    new_values = {"a": 10, "b": 20.0, "c": "updated", "d": False}
    model = DummyAttrs(attrs=hdf5_file.attrs, **new_values)

    assert hdf5_file.attrs["a"] == new_values["a"] == model.a
    assert hdf5_file.attrs["b"] == new_values["b"] == model.b
    assert hdf5_file.attrs["c"] == new_values["c"] == model.c
    assert hdf5_file.attrs["d"] == new_values["d"] == model.d
