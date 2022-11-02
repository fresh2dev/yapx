__all__ = [
    "__version__",
    "ArgumentParser",
    "run",
    "run_command",
    "types",
    "actions",
    "utils",
    "exceptions",
    "arg",
    "argparse_action",
]

import os

from . import actions, exceptions, types, utils
from .arg import arg
from .argparse_action import argparse_action
from .argument_parser import ArgumentParser, run, run_command

with open(
    os.path.join(os.path.dirname(__file__), "VERSION"), mode="r", encoding="utf-8"
) as f:
    __version__: str = f.read().strip()
