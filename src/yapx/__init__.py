from contextlib import suppress

from . import actions, exceptions, types, utils
from .__version__ import __version__
from .arg import arg
from .argparse_action import argparse_action
from .argument_parser import ArgumentParser, run, run_commands

with suppress(ModuleNotFoundError):
    from rich.traceback import install

    install(show_locals=True, extra_lines=3)


__all__ = [
    "__version__",
    "ArgumentParser",
    "run",
    "run_commands",
    "types",
    "actions",
    "utils",
    "exceptions",
    "arg",
    "argparse_action",
]
