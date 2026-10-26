import inspect
from dataclasses import _MISSING_TYPE
from types import NoneType, UnionType
from typing import Any, ClassVar, Union, get_args, get_origin

import numpy as np


def extract_annotations(field_annotation: type[Any]) -> list[type[Any]]:
    args: tuple[type[Any], ...]
    origin = get_origin(field_annotation)
    if origin is Union or origin is UnionType:
        args = get_args(field_annotation) 
    else:
        args = (field_annotation,)

    return [
        arg
        for arg in args
        if (arg is not NoneType) and (arg is not _MISSING_TYPE)
    ]


def extract_single_annotation(field_annotation: type[Any]) -> type[Any]:
    annotations = extract_annotations(field_annotation)
    n = len(annotations)
    if n != 1:
        raise ValueError(
            f"Expected exactly one annotation, got {n}: {annotations}"
        )
    return annotations[0]


def extract_annotation_class(field_annotation: type[Any]) -> type[Any]:
    annotation_cls = extract_single_annotation(field_annotation)
    if not inspect.isclass(annotation_cls):
        raise TypeError(f"{annotation_cls} is not a class type")

    return annotation_cls


def is_classvar(field_annotation: type[Any]) -> bool:
    origin = get_origin(field_annotation)
    return origin is ClassVar


def is_optional(field_annotation: type[Any]) -> bool:
    origin = get_origin(field_annotation)
    if origin is Union or origin is UnionType:
        args = get_args(field_annotation)
        return NoneType in args
    return False


def shape_is_exact(shape_annotation: type[Any]) -> bool:
    """
    Returns True if the given shape annotation represents an exact tuple type.
    """
    return get_origin(shape_annotation) is tuple


def dtype_is_exact(dtype_annotation: type[Any]) -> bool:
    """
    Returns True if the given dtype annotation represents an exact numpy dtype, 
    and it can be used to create a numpy dtype instance.
    """
    if (get_origin(dtype_annotation) is None) \
        and dtype_annotation is not np.dtype:
        try:
            np.dtype(dtype_annotation)
            return True
        except TypeError:
            return False
    return False