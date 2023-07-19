from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import exceptions, types
from .__version__ import __version__
from .arg import arg
from .argument_parser import ArgumentParser
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
    "arg",
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
    subcommands: Optional[Sequence[Callable[..., Any]]] = None,
    named_subcommands: Optional[Dict[str, Callable[..., Any]]] = None,
    **kwargs: Any,
) -> Any:
    """Use given functions to construct an ArgumentParser.

    Args:
        command: the root command function
        subcommands: a list of subcommand functions
        named_subcommands: a dict of named subcommand functions
        **kwargs: passed to the ArgumentParser constructor
    """
    # pylint: disable=protected-access
    return ArgumentParser._build_parser(
        command=command,
        subcommands=subcommands,
        named_subcommands=named_subcommands,
        **kwargs,
    )


def run(
    command: Optional[Callable[..., Any]] = None,
    subcommands: Optional[Sequence[Callable[..., Any]]] = None,
    named_subcommands: Optional[Dict[str, Callable[..., Any]]] = None,
    args: Optional[List[str]] = None,
    default_args: Optional[List[str]] = None,
    **kwargs: Any,
) -> Any:
    """Use given functions to construct an ArgumentParser,
    parse the args, and invoke the appropriate command.


    Args:
        command: the root command function
        subcommands: a list of subcommand functions
        named_subcommands: a dict of named subcommand functions
        args: arguments to parse (default=`sys.argv[1:]`)
        default_args: arguments to parse when no arguments are given.
        **kwargs: passed to the ArgumentParser constructor

    Examples:
        >>> import yapx
        ...
        >>> def print_nums(*args):
        ...     print('Args: ', *args)
        ...
        >>> def find_evens(*args):
        ...     return [x for x in args if int(x) % 2 == 0]
        ...
        >>> def find_odds(*args):
        ...     return [x for x in args if int(x) % 2 != 0]
        ...
        >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
        >>> yapx.run(print_nums, [find_evens, find_odds], args=cli_args)
        Args:  1 2 3 4 5
        ['1', '3', '5']
    """
    # pylint: disable=protected-access
    with suppress(SystemExit):
        return ArgumentParser._run(
            command=command,
            subcommands=subcommands,
            named_subcommands=named_subcommands,
            args=args,
            default_args=default_args,
            **kwargs,
        )


def run_commands(
    *parser_args: Any,
    **parser_kwargs: Any,
) -> Any:
    """Use given functions to construct an ArgumentParser,
    parse the args, and invoke the appropriate command.

    `yapx.run_commands(...)` is equivalent to `yapx.run(None, ...)`, to be used when
    there is no root command.

    Examples:
        >>> import yapx
        ...
        >>> def find_evens(*args):
        ...     return [x for x in args if int(x) % 2 == 0]
        ...
        >>> def find_odds(*args):
        ...     return [x for x in args if int(x) % 2 != 0]
        ...
        >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
        >>> yapx.run_commands([find_evens, find_odds], args=cli_args)
        ['1', '3', '5']
    """
    # pylint: disable=protected-access
    return run(None, *parser_args, **parser_kwargs)


def run_patched(
    *args: Any,
    test_args: Optional[List[str]] = None,
    disable_pydantic: bool = False,
    **kwargs: Any,
) -> Any:
    """Same as `yapx.run`, with the ability to patch args and disable pydantic.

    Args:
        test_args: patch sys.argv with these args
        disable_pydantic: disable the use of pydantic for additional validation

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
