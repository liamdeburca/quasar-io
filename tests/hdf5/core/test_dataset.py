import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from quasar_io.hdf5.core import BaseAttrs, BaseFloatDataset


class Dummy1DFloatDataset(BaseFloatDataset):
    shape: tuple[int] | None = None


class Dummy2DFloatDataset(BaseFloatDataset):
    shape: tuple[int, int] | None = None


class DummyAttrs(BaseAttrs):
    # Required fields
    a: int | None = None
    b: float | None = None
    c: str | None = None
    d: bool | None = None


class Dummy1DFloatDatasetWithAttrs(Dummy1DFloatDataset):
    attrs: DummyAttrs | None = None


floats_1d = np.empty((10,), dtype=np.float32)
floats_2d = np.empty((10, 5), dtype=np.float32)

ints_1d = np.empty((10,), dtype=np.int32)
ints_2d = np.empty((10, 5), dtype=np.int32)

def test_default_initialization_1d():
    d = Dummy1DFloatDataset(data=floats_1d)

    assert d.dataset is None
    assert d.attrs is None
    assert d.shape == (10,)
    assert d.dtype == np.float32
    assert np.array_equal(d.data, floats_1d)

def test_default_initialization_2d():
    d = Dummy2DFloatDataset(data=floats_2d)
    
    assert d.dataset is None
    assert d.attrs is None
    assert d.shape == (10, 5)
    assert d.dtype == np.float32
    assert np.array_equal(d.data, floats_2d)

def test_default_initialization_fails():
    with pytest.raises((TypeError, ValidationError)):
        Dummy1DFloatDataset(data=floats_2d)
        Dummy1DFloatDataset(data=ints_1d)
        Dummy1DFloatDataset(data=ints_2d)

        Dummy2DFloatDataset(data=floats_1d)
        Dummy2DFloatDataset(data=ints_1d)
        Dummy2DFloatDataset(data=ints_2d)

def test_initialization_with_just_array():
    Dummy1DFloatDataset.model_validate(floats_1d)
    Dummy2DFloatDataset.model_validate(floats_2d)

def test_dataset_initialisation(hdf5_file: h5py.File):
    dataset_1d = hdf5_file.create_dataset(
        "dummy_1d",
        data=floats_1d
    )
    d1 = Dummy1DFloatDataset(dataset=dataset_1d)
    dataset_2d = hdf5_file.create_dataset(
        "dummy_2d",
        data=floats_2d
    )
    d2 = Dummy2DFloatDataset(dataset=dataset_2d)

    assert d1.dataset is dataset_1d
    assert d1.attrs.attrs == dataset_1d.attrs
    assert d1.shape == dataset_1d.shape
    assert d1.dtype == dataset_1d.dtype
    assert np.array_equal(d1.data, floats_1d)

    assert d2.dataset is dataset_2d
    assert d2.attrs.attrs == dataset_2d.attrs
    assert d2.shape == dataset_2d.shape
    assert d2.dtype == dataset_2d.dtype
    assert np.array_equal(d2.data, floats_2d)

def test_dataset_to_file(hdf5_file: h5py.File):
    d1 = Dummy1DFloatDataset(data=floats_1d)
    d2 = Dummy2DFloatDataset(data=floats_2d)

    dataset_1d = d1.toHDF5(hdf5_file, "dummy_1d")
    dataset_2d = d2.toHDF5(hdf5_file, "dummy_2d")

    assert d1.dataset is dataset_1d
    assert d1.shape == dataset_1d.shape
    assert d1.dtype == dataset_1d.dtype
    assert np.array_equal(d1.data, dataset_1d[()])

    assert d2.dataset is dataset_2d
    assert d2.shape == dataset_2d.shape
    assert d2.dtype == dataset_2d.dtype
    assert np.array_equal(d2.data, dataset_2d[()])

    t1 = Dummy1DFloatDataset(dataset=dataset_1d)
    t2 = Dummy2DFloatDataset(dataset=dataset_2d)

    assert d1.dataset is t1.dataset
    assert d1.shape == t1.shape
    assert d1.dtype == t1.dtype
    assert np.array_equal(d1.data, t1.data)

    assert d2.dataset is t2.dataset
    assert d2.shape == t2.shape
    assert d2.dtype == t2.dtype
    assert np.array_equal(d2.data, t2.data)

def test_initialization_with_attrs():
    Dummy1DFloatDataset.model_validate({
        "data": floats_1d,
        "attrs": {
            "a": 1,
            "b": 2.0,
            "c": "test",
            "d": True
        }
    })

def test_initialization_with_attrs_fails():
    with pytest.raises(ValidationError):
        Dummy1DFloatDataset.model_validate({
            "data": floats_2d, # Wrong array dimensionality
            "attrs": {
                "a": 1,
                "b": 2.0,
                "c": "test",
                "d": True
            }
        })
    
    with pytest.raises(ValidationError):
        Dummy1DFloatDataset.model_validate({
            "data": floats_2d,
            "attrs": {
                "a": 1,
                "b": 2.0,
                "c": "test",
                "d": True,
                "e": "extra arg", # Some extra argument
            }
        })