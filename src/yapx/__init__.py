from contextlib import suppress

from . import exceptions, types
from .__version__ import __version__
from .arg import arg
from .argument_parser import ArgumentParser, run, run_commands

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
    "run",
    "run_commands",
    "types",
    "exceptions",
    "arg",
]
