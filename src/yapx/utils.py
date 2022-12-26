import dataclasses
from typing import Any, Type

# pylint: disable=unused-import
from .arg import (
    convert_to_command_string,
    convert_to_flag_string,
    make_dataclass_from_func,
)
from .types import Dataclass

__all__ = [
    "convert_to_command_string",
    "convert_to_flag_string",
    "make_dataclass_from_func",
    "str2bool",
    "is_dataclass_type",
    "coalesce",
]


try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard


def str2bool(string: str) -> bool:
    return string.lower() in ("1", "true", "t", "yes", "y")


def is_dataclass_type(candidate: Any) -> TypeGuard[Type[Dataclass]]:
    return dataclasses.is_dataclass(candidate)


def is_instance(candidate: Any, test_type: Type[Any]) -> bool:
    try:
        return isinstance(candidate, test_type)
    except TypeError:
        return False


def is_subclass(candidate: Any, test_type: Type[Any]) -> bool:
    try:
        return issubclass(candidate, test_type)
    except TypeError:
        return False


def coalesce(x: Any, d: Any, null_or_empty: bool = False) -> Any:
    if (null_or_empty and x) or (not null_or_empty and x is not None):
        return x
    return d
