from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List, Optional

from .types import ArgumentParser


@dataclass(frozen=True)
class Context:
    """An immutable object that is passed to arguments annotated with `yapx.Context`,
    giving them access to the argument parser, raw arguments, parsed namespace, and any relay value.

    Attributes:
        parser: the root argparse ArgumentParser
        subparser: this command's subparser
        args: raw command-line arguments
        namespace: parsed command-line arguments
        relay_value: Any value returned from the root command

    Examples:
        >>> import yapx
        ...
        >>> def print_nums(*args: int):
        ...     print('Args: ', *args)
        ...     return args
        ...
        >>> def find_evens(_context: yapx.Context):
        ...     return [x for x in _context.relay_value if x % 2 == 0]
        ...
        >>> def find_odds(_context: yapx.Context):
        ...     return [x for x in _context.relay_value if x % 2 != 0]
        ...
        >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
        >>> yapx.run(print_nums, [find_evens, find_odds], args=cli_args)
        Args:  1 2 3 4 5
        [1, 3, 5]
    """

    parser: ArgumentParser
    subparser: Optional[ArgumentParser]
    args: List[str]
    namespace: Namespace
    relay_value: Any
