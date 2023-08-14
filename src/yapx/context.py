from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List

from .types import ArgumentParser


@dataclass(frozen=True)
class Context:
    """An immutable object that is passed to arguments annotated with `yapx.Context`,
    giving them access to the argument parser, raw arguments, parsed namespace, and any relay value.

    Attributes:
        parser: the argparse ArgumentParser
        args: raw command-line arguments
        namespace: parsed command-line arguments
        relay_value: Any value returned from the root command

    Examples:
        >>> import yapx
        ...
        >>> def print_nums(*args):
        ...     print('Args: ', *args)
        ...     return args
        ...
        >>> def find_evens(_context: yapx.Context):
        ...     return [x for x in _context.relay_value if int(x) % 2 == 0]
        ...
        >>> def find_odds(_context: yapx.Context):
        ...     return [x for x in _context.relay_value if int(x) % 2 != 0]
        ...
        >>> cli_args = ['1', '2', '3', '4', '5', 'find-odds']
        >>> yapx.run(print_nums, [find_evens, find_odds], args=cli_args)
        Args:  1 2 3 4 5
        ['1', '3', '5']
    """

    parser: ArgumentParser
    args: List[str]
    namespace: Namespace
    relay_value: Any
