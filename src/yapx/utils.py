import sys
from contextlib import suppress
from dataclasses import is_dataclass
from enum import Enum
from functools import wraps
from typing import Any, Optional, Type, TypeVar, Union

# pylint: disable=unused-import
from .arg import (
    convert_to_command_string,
    convert_to_flag_string,
    make_dataclass_from_func,
)
from .exceptions import raise_unsupported_type_error
from .types import Dataclass

__all__ = [
    "add_argument_to",
    "convert_to_command_string",
    "convert_to_flag_string",
    "make_dataclass_from_func",
    "create_pydantic_model_from_dataclass",
    "is_dataclass_type",
    "is_pydantic_available",
    "is_shtab_available",
    "cast_bool",
    "cast_type",
]

try:
    from shtab import add_argument_to
except ModuleNotFoundError:

    def add_argument_to():
        ...


try:
    from pydantic import ValidationError, parse_obj_as
    from pydantic.dataclasses import create_pydantic_model_from_dataclass
except ModuleNotFoundError:

    def parse_obj_as():
        ...

    def create_pydantic_model_from_dataclass():
        ...

    class ValidationError(Exception):
        ...


try:
    from trogon.argparse import add_tui_argument
except ModuleNotFoundError:

    def add_tui_argument():
        ...


try:
    from rich_argparse import RawTextRichHelpFormatter as RawTextHelpFormatter
except ModuleNotFoundError:
    from argparse import RawTextHelpFormatter  # noqa: F401

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

T = TypeVar("T", bound=type)


def is_dataclass_type(candidate: Any) -> TypeGuard[Type[Dataclass]]:
    return is_dataclass(candidate)


@wraps(isinstance)
def try_isinstance(*args: Any, **kwargs: Any) -> bool:
    with suppress(TypeError):
        return isinstance(*args, **kwargs)
    return False


@wraps(issubclass)
def try_issubclass(*args: Any, **kwargs: Any) -> bool:
    with suppress(TypeError):
        return issubclass(*args, **kwargs)
    return False


def coalesce(x: Any, d: Any, null_or_empty: bool = False) -> Any:
    if (null_or_empty and x) or (not null_or_empty and x is not None):
        return x
    return d


def is_pydantic_available() -> bool:
    return bool(create_pydantic_model_from_dataclass.__module__ != __name__)


def is_shtab_available() -> bool:
    return bool(add_argument_to.__module__ != __name__)


def is_tui_available() -> bool:
    return bool(add_tui_argument.__module__ != __name__)


def cast_bool(value: Union[None, str, bool]) -> bool:
    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        return bool(value)

    value_lower: str = value.lower()

    if value_lower in ("1", "true", "t", "yes", "y", "on"):
        return True

    if value_lower in ("0", "false", "f", "no", "n", "off"):
        return False

    raise ValueError(f"Invalid literal for bool(): {value}")


def cast_type(target_type: Optional[T], value: Any) -> Optional[T]:
    if value is None or target_type is None or try_isinstance(value, target_type):
        return value

    if target_type is bool:
        return cast_bool(value)

    if try_issubclass(target_type, Enum):
        return target_type[value]

    try:
        return target_type(value)
    except TypeError as e:
        if is_pydantic_available():
            return parse_obj_as(target_type, value)

        raise_unsupported_type_error(type(value), from_exception=e)
