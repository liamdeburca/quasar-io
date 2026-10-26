import h5py
import numpy as np
import pytest

from quasar_io.hdf5.core import (
    BaseAttrs,
    BaseFloatDataset,
    BaseGroup,
    OptionalBaseFloatDataset,
)


class Dummy1DFloatDataset(BaseFloatDataset):
    shape: tuple[int] | None = None


class Dummy2DFloatDataset(BaseFloatDataset):
    shape: tuple[int, int] | None = None


class OptionalDummy1DFloatDataset(OptionalBaseFloatDataset):
    shape: tuple[int] | None = None


class DummyAttrs(BaseAttrs):
    # Required fields
    a: int | None = None
    b: float | None = None
    c: str | None = None
    d: bool | None = None


class DummyGroup(BaseGroup):
    attrs: DummyAttrs | None = None
    arr: Dummy1DFloatDataset | None = None
    mat: Dummy2DFloatDataset | None = None

    optional_arr: OptionalDummy1DFloatDataset | None = None


floats_1d = np.empty((10,), dtype=np.float32)
floats_2d = np.empty((10, 5), dtype=np.float32)

ints_1d = np.empty((10,), dtype=np.int32)
ints_2d = np.empty((10, 5), dtype=np.int32)

# Common attrs used across tests
DEFAULT_ATTRS = {"a": 1, "b": 2.0, "c": "t", "d": True}


def test_initialization():
    group = DummyGroup(
        arr=floats_1d,
        mat=floats_2d,
    )
    # Values provided as raw arrays should be accepted
    assert group.arr is not None
    assert group.mat is not None
    assert group.arr.shape == (10,)
    assert group.mat.shape == (10, 5)
    assert np.array_equal(group.arr.data, floats_1d)
    assert np.array_equal(group.mat.data, floats_2d)
    # optional field not provided should be None
    assert group.optional_arr is None


def test_missing_required_datasets_raises():
    # Both datasets are required on DummyGroup; omitting them should fail
    with pytest.raises(ValueError):
        DummyGroup()


def test_group_population_from_hdf5_file(hdf5_file: h5py.File):
    # Create datasets and attrs in the file
    hdf5_file.create_dataset("arr", data=floats_1d)
    hdf5_file.create_dataset("mat", data=floats_2d)
    hdf5_file.attrs.update(DEFAULT_ATTRS)

    g = DummyGroup(group=hdf5_file)
    # attrs populated
    assert isinstance(g.attrs, DummyAttrs)
    assert g.attrs.a == 1
    # datasets populated
    assert g.arr.dataset.name.endswith("arr")
    assert g.mat.dataset.name.endswith("mat")
    assert g.arr.shape == (10,)
    assert g.mat.shape == (10, 5)
    # optional dataset not present -> optional_arr remains None
    assert g.optional_arr is None


def test_optional_dataset_handling(hdf5_file: h5py.File):
    # Define a group model with an optional dataset field
    class OptGroup(BaseGroup):
        attrs: DummyAttrs | None = None
        arr: OptionalBaseFloatDataset | None = None
        mat: Dummy2DFloatDataset | None = None

    # Only create the 2d dataset
    hdf5_file.create_dataset("mat", data=floats_2d)
    # Provide required attributes so BaseGroup can populate DummyAttrs
    hdf5_file.attrs.update(DEFAULT_ATTRS)

    # Should not raise even though `arr` is optional and missing
    g = OptGroup(group=hdf5_file)
    assert g.arr is None
    assert g.mat is not None
    # optional_arr on DummyGroup should not be touched by OptGroup
    assert not hasattr(g, "optional_arr") or g.arr is None


def test_optional_dataset_populated_when_present(hdf5_file: h5py.File):
    # Create only the optional dataset in the file and ensure it populates
    hdf5_file.create_dataset("optional_arr", data=floats_1d)
    hdf5_file.create_dataset("arr", data=floats_1d)
    hdf5_file.create_dataset("mat", data=floats_2d)
    hdf5_file.attrs.update(DEFAULT_ATTRS)

    g = DummyGroup(group=hdf5_file)
    # optional_arr should now be populated
    assert g.optional_arr is not None
    assert g.optional_arr.shape == (10,)
    assert np.array_equal(g.optional_arr.data, floats_1d)


def test_type_mismatch_in_dataset_raises(hdf5_file: h5py.File):
    # Create an integer dataset where a float dataset is expected
    hdf5_file.create_dataset("arr", data=ints_1d)
    hdf5_file.create_dataset("mat", data=floats_2d)
    # Provide required attributes so BaseGroup proceeds to dataset checks
    hdf5_file.attrs.update(DEFAULT_ATTRS)
    with pytest.raises(TypeError):
        DummyGroup(group=hdf5_file)


def test_missing_required_group_raises(hdf5_file: h5py.File):
    # Parent group expects a subgroup `child` which won't exist
    class Parent(BaseGroup):
        attrs: DummyAttrs | None = None
        child: DummyGroup | None = None

    # No subgroup created -> constructing Parent with the file should raise
    # Ensure required attrs exist so group-level validation proceeds
    hdf5_file.attrs.update(DEFAULT_ATTRS)
    with pytest.raises(ValueError):
        Parent(group=hdf5_file)


def test_nested_group_population(hdf5_file: h5py.File):
    # Create a subgroup with required datasets and attach to file
    sub = hdf5_file.create_group("child")
    sub.create_dataset("arr", data=floats_1d)
    sub.create_dataset("mat", data=floats_2d)
    # Provide attributes for the subgroup so DummyAttrs can populate
    sub.attrs.update({"a": 1, "b": 2.0, "c": "t", "d": True})

    class Parent(BaseGroup):
        attrs: DummyAttrs | None = None
        child: DummyGroup | None = None

    # Ensure file-level attrs exist
    hdf5_file.attrs.update(DEFAULT_ATTRS)

    p = Parent(group=hdf5_file)
    assert p.child is not None
    assert p.child.arr is not None
    assert p.child.mat is not None


def test_constructor_accepts_file_positional(hdf5_file: h5py.File):
    # Create required datasets and attrs
    hdf5_file.create_dataset("arr", data=floats_1d)
    hdf5_file.create_dataset("mat", data=floats_2d)
    hdf5_file.attrs.update(DEFAULT_ATTRS)

    # Pass the file object positionally to model_validate; coerce_to_group
    # should wrap it into {'group': hdf5_file} when using model_validate.
    g = DummyGroup.model_validate(hdf5_file)
    assert isinstance(g.attrs, DummyAttrs)
    assert g.arr is not None
    assert g.mat is not None


def test_constructor_rejects_non_h5py_objects():
    # Passing a non-h5py object should ultimately raise a TypeError
    with pytest.raises(TypeError):
        DummyGroup(group="not_a_group_or_file")
