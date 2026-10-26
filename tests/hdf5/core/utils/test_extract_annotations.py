from quasar_io.hdf5.core.utils import extract_annotations


class DummyA:
    pass


class DummyB:
    pass


def test_extract_annotations_with_pipe_and_none():
    # DummyA | None should return only DummyA
    annotations = extract_annotations(DummyA | None)
    assert annotations == [DummyA]


def test_extract_annotations_multiple_types_pipe():
    # DummyA | DummyB should return both classes
    annotations = extract_annotations(DummyA | DummyB)
    assert set(annotations) == {DummyA, DummyB}


def test_extract_annotations_filters_out_missing_and_none():
    # list[int] | None should return list[int] only
    annotations = extract_annotations(list[int] | None)
    assert annotations == [list[int]]
