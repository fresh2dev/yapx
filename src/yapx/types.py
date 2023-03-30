import sys
from enum import Enum
from typing import Any, Dict, Optional, Sequence, Union

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=unused-import # noqa: F401
else:
    from typing_extensions import Literal  # pylint: disable=unused-import # noqa: F401

if sys.version_info >= (3, 9):
    from typing import Annotated  # pylint: disable=unused-import # noqa: F401
else:
    from typing_extensions import (  # pylint: disable=unused-import # noqa: F401
        Annotated,
    )

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

NoneType = type(None)

ArgValueType = Union[None, str, Sequence[str]]

__all__ = [
    "Dataclass",
    "ArgumentParser",
    "NoneType",
    "ArgValueType",
    "Literal",
    "Enum",
    "Annotated",
    "Protocol",
]


class Dataclass(Protocol):
    __dataclass_fields__: Any


class ArgumentParser(Protocol):
    kv_separator = str

    _mutually_exclusive_args: Dict[str, Dict[str, Optional[str]]]

    _inner_type_conversions: Dict[str, type]

    def print_help(self) -> None:
        ...

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        ...
