from contextlib import suppress

from . import exceptions, types
from .__version__ import __version__
from .arg import arg
from .argument_parser import ArgumentParser, build_parser, run, run_commands
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
    "types",
]
