from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Union

from .utils import convert_to_command_string


@dataclass
class Command:
    function: Callable[..., Any]
    name: str
    kwargs: Dict[str, Any]

    def __post_init__(self):
        if not self.name and self.function is None:
            raise ValueError("Must specify either 'name' or 'function'.")

        if not self.name:
            assert self.function is not None
            if not self.function.__name__:
                raise ValueError("Must specify 'name' when function is nameless.")
            self.name = convert_to_command_string(self.function.__name__)
        elif self.function is None:
            self.function = lambda: ...  # noqa

    def __hash__(self):
        return hash(self.name)


def cmd(
    function: Union[None, Command, Callable[..., Any]] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Command:
    """Modify properties of a subcommand.

    Args:
        function: the function called when this subcommand is provided.
        name: the name of the subcommand added to the parser.
        **kwargs: passed to the `ArgumentParser` constructor for this command.

    Returns:
        ...

    Examples:
        >>> import yapx
        ...
        >>> def sum_ints(*args: int):
        ...     print(sum(args))
        ...
        >>> yapx.run_commands([
        ...     yapx.cmd(sum_ints, name="add-nums")
        ... ], args=["add-nums", "1", "2", "3"])
        6
    """
    if isinstance(function, Command):
        return function

    return Command(function=function, name=name, kwargs=kwargs)


CommandOrCallable = Union[Command, Callable[..., Any]]
CommandSequence = Sequence[CommandOrCallable]
CommandMap = Dict[Optional[CommandOrCallable], Union["CommandMap", CommandSequence]]
