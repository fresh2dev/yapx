from contextlib import suppress
from typing import Any, List, Optional

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


def build_parser(*args: Any, **kwargs: Any) -> Any:
    # pylint: disable=protected-access
    return ArgumentParser._build_parser(*args, **kwargs)


def run(*args: Any, **kwargs: Any) -> Any:
    """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

    Args:
        *parser_args:
        args:
        default_args:
        **parser_kwargs:

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
        return ArgumentParser._run(*args, **kwargs)


def run_commands(*args: Any, **kwargs: Any) -> Any:
    """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

    Args:
        *parser_args:
        args:
        default_args:
        **parser_kwargs:

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
    return run(None, *args, **kwargs)


def run_patched(
    *args: Any,
    test_args: Optional[List[str]] = None,
    disable_pydantic: bool = False,
    **kwargs: Any,
) -> Any:
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
