from argparse import Namespace
from dataclasses import dataclass
from typing import Any, List

from .types import ArgumentParser


@dataclass(frozen=True)
class Context:
    parser: ArgumentParser
    args: List[str]
    namespace: Namespace
    relay_value: Any
