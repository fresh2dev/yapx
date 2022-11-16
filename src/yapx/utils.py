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
]


try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard


def str2bool(string: str) -> bool:
    return string.lower() in ("1", "true", "t", "yes", "y")


def is_dataclass_type(candidate: Any) -> TypeGuard[Type[Dataclass]]:
    return dataclasses.is_dataclass(candidate)
