import sys
from typing import Any, Dict, Optional, Sequence, Union

__all__ = ["Dataclass", "ArgumentParser", "NoneType", "ArgValueType"]


if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

NoneType = type(None)

ArgValueType = Union[None, str, Sequence[str]]


class Dataclass(Protocol):
    __dataclass_fields__: Any


class ArgumentParser(Protocol):
    kv_separator = str

    _inner_type_conversions: Dict[str, type]

    def print_help(self) -> None:
        ...

    def exit(self, status: int = 0, message: Optional[str] = None) -> None:
        ...
