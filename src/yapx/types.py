__all__ = ["Dataclass", "ArgumentParser", "NoneType", "ArgValueType"]

import sys
from typing import Any, Dict, Optional, Sequence, Union

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

NoneType = type(None)

ArgValueType = Union[None, str, Sequence[str]]


class Dataclass(Protocol):
    __dataclass_fields__: Any


class ArgumentParser(Protocol):
    _inner_type_conversions: Dict[str, type]

    def print_help(self) -> None:
        ...

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        ...
