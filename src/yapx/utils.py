import dataclasses
import sys
from contextlib import suppress
from typing import Any, Dict, List, Tuple, Type, Union

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


if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard


def str2bool(value: Union[str, bool]) -> bool:
    if isinstance(value, bool):
        return value

    value_lower: str = value.lower()

    if value_lower in ("1", "true", "t", "yes", "y"):
        return True

    if value_lower in ("0", "false", "f", "no", "n"):
        return False

    raise ValueError(f"Invalid literal for bool(): {value}")


def is_dataclass_type(candidate: Any) -> TypeGuard[Type[Dataclass]]:
    return dataclasses.is_dataclass(candidate)


def try_isinstance(candidate: Any, test_type: Type[Any]) -> bool:
    with suppress(TypeError):
        return isinstance(candidate, test_type)
    return False


def try_issubclass(candidate: Any, test_type: Type[Any]) -> bool:
    with suppress(TypeError):
        return issubclass(candidate, test_type)
    return False


def coalesce(x: Any, d: Any, null_or_empty: bool = False) -> Any:
    if (null_or_empty and x) or (not null_or_empty and x is not None):
        return x
    return d


def parse_sequence(
    *args: str,
    kv_separator: str = "=",
) -> Tuple[List[str], Dict[str, str]]:
    parsed_args: List[str] = []
    parsed_kwargs: Dict[str, str] = {}

    for x in args:
        if x and kv_separator in x:
            k, v = x.split(kv_separator, 1)
            parsed_kwargs[k.strip()] = v.strip() if v.strip() else None
        else:
            parsed_args.append(x)

    return parsed_args, parsed_kwargs
