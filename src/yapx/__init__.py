from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from . import exceptions, types
from .__version__ import __version__
from .arg import arg, counting_arg, feature_arg, unbounded_arg
from .argument_parser import ArgumentParser
from .command import Command, CommandMap, CommandOrCallable, CommandSequence, cmd
from .context import Context
from .namespace import Namespace
from .utils import is_pydantic_available, is_shtab_available, is_tui_available

with suppress(ModuleNotFoundError):
    import os

    from rich.traceback import install

    if os.getenv("YAPX_DEBUG", None) == "1":
        install(show_locals=True, extra_lines=5)
    else:
        install(show_locals=False, extra_lines=3)


__all__ = [
    "__version__",
    "ArgumentParser",
    "Namespace",
    "cmd",
    "Command",
    "arg",
    "counting_arg",
    "feature_arg",
    "unbounded_arg",
    "build_parser",
    "exceptions",
    "is_pydantic_available",
    "is_shtab_available",
    "is_tui_available",
    "run",
    "run_commands",
    "run_patched",
    "run_commands_patched",
    "types",
]


def build_parser(
    command: Optional[Callable[..., Any]] = None,
    subcommands: Union[
        None,
        str,
        CommandOrCallable,
        CommandSequence,
        CommandMap,
    ] = None,
    **kwargs: Any,
) -> ArgumentParser:
    """Use given functions to construct an ArgumentParser.

    Args:
        command: the root command function
        subcommands: a list or mapping of subcommand functions
        **kwargs: passed to the ArgumentParser constructor

    Returns:
        ...

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
        >>> parser = yapx.build_parser(print_nums, [find_evens, find_odds])
        ...
        >>> import argparse
        >>> isinstance(parser, argparse.ArgumentParser)
        True
    """
    # pylint: disable=protected-access
    return ArgumentParser._build_parser(
        command=command,
        subcommands=subcommands,
        **kwargs,
    )


def run(
    command: Optional[Callable[..., Any]] = None,
    subcommands: Union[
        None,
        str,
        CommandOrCallable,
        CommandSequence,
        CommandMap,
    ] = None,
    args: Optional[List[str]] = None,
    default_args: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Use given functions to construct an ArgumentParser,
    parse the args, invoke the appropriate command, and return any result.

    Args:
        command: the root command function
        subcommands: a list or mapping of subcommand functions
        args: arguments to parse (default=`sys.argv[1:]`)
        default_args: arguments to parse when no arguments are given.
        **kwargs: passed to the ArgumentParser constructor

    Returns:
        ...

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
    # pylint: disable=protected-access
    return ArgumentParser._run(
        command=command,
        subcommands=subcommands,
        args=args,
        default_args=default_args,
        **kwargs,
    )


def run_commands(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Use given functions to construct an ArgumentParser,
    parse the args, invoke the appropriate command, and return any result.

    `yapx.run_commands(...)` is equivalent to `yapx.run(None, ...)`, to be used when
    there is no root command.

    Args:
        *args: ...
        **kwargs: ...

    Returns:
        ...

    Examples:
        >>> import yapx
        ...
        >>> def find_evens(*args: int):
        ...     return [x for x in args if x % 2 == 0]
        ...
        >>> def find_odds(*args: int):
        ...     return [x for x in args if x % 2 != 0]
        ...
        >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
        >>> yapx.run_commands([find_evens, find_odds], args=cli_args)
        [1, 3, 5]
    """
    # pylint: disable=protected-access
    return run(None, *args, **kwargs)


def run_patched(
    *args: Any,
    test_args: Optional[List[str]] = None,
    disable_pydantic: bool = False,
    **kwargs: Any,
) -> Any:
    """For use in tests. Same as `yapx.run`, with the ability to patch args and disable pydantic.

    Args:
        *args: ...
        test_args: patch sys.argv with these args
        disable_pydantic: disable the use of pydantic for additional validation
        **kwargs: ...

    Returns:
        ...
    """
    from unittest import mock

    from .argument_parser import sys as _sys

    if not test_args:
        test_args = []

    pydantic_patch: Optional[mock._patch] = None

    try:
        if disable_pydantic:
            pydantic_patch = mock.patch(
                "yapx.argument_parser.is_pydantic_available",
                False,
            )
            pydantic_patch.start()

        with mock.patch.object(_sys, "argv", [_sys.argv[0], *test_args]):
            return run(*args, **kwargs)
    finally:
        if pydantic_patch:
            pydantic_patch.stop()


def run_commands_patched(
    *args: Any,
    **kwargs: Any,
) -> Any:
    return run_patched(None, *args, **kwargs)
