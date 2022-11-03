__all__ = ["arg"]

import collections.abc
import inspect
import os
from argparse import Action
from dataclasses import MISSING, Field, dataclass, field, make_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

from .types import Dataclass

try:
    from typing import get_type_hints
except ImportError:
    from typing_extensions import get_type_hints

ARGPARSE_ARG_METADATA_KEY: str = "_argparse_argument"


@dataclass
class ArgparseArg:
    dest: Optional[str] = None
    option_strings: Optional[Sequence[str]] = None
    type: Union[None, Type[Any], Callable[[str], Any]] = None
    action: Union[None, str, Type[Action]] = None
    required: bool = True
    group: Optional[str] = None
    exclusive: Optional[bool] = False
    nargs: Optional[str] = None
    const: Optional[Any] = None
    default: Optional[Any] = None
    choices: Optional[Sequence[Any]] = None
    help: Optional[str] = None
    metavar: Optional[str] = None
    pos: Optional[bool] = False
    _env_var: Optional[str] = None

    def __post_init__(self) -> None:
        if self.dest and not self.option_strings:
            self.option_strings = [convert_to_flag_string(self.dest)]

    def asdict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def arg(
    default: Optional[Any] = MISSING,
    env_var: Optional[str] = None,
    pos: Optional[bool] = False,
    group: Optional[str] = None,
    exclusive: Optional[bool] = False,
    flags: Optional[Sequence[str]] = None,
    # pylint: disable=redefined-builtin
    help: Optional[str] = None,
    metavar: Optional[str] = None,
    action: Optional[Action] = None,
) -> Field:  # type: ignore
    if env_var:
        default_from_env = os.getenv(env_var, None)
        if default_from_env:
            default = default_from_env
        else:
            default_from_file = os.getenv(env_var + "_FILE", None)
            if default_from_file and os.path.exists(default_from_file):
                with open(default_from_file, mode="r", encoding="utf8") as f:
                    default = f.read().strip()

    required = default is MISSING

    metadata: Dict[str, ArgparseArg] = {
        ARGPARSE_ARG_METADATA_KEY: ArgparseArg(
            action=action,
            pos=pos,
            option_strings=flags,
            required=required,
            group=group,
            exclusive=exclusive,
            default=(None if required else default() if callable(default) else default),
            help=help,
            metavar=metavar,
            _env_var=env_var,
        )
    }

    kwargs: Dict[str, Any] = {"metadata": metadata}

    default_param: str = "default_factory" if callable(default) else "default"
    kwargs[default_param] = default

    fld: Field = field(**kwargs)  # type: ignore
    assert isinstance(fld, Field)
    return fld


def convert_to_command_string(x: str) -> str:
    x = x.strip()

    x_prefix: str = "x_"
    under: str = "_"
    if x.startswith(x_prefix):
        x = x[len(x_prefix) :]
    elif x.startswith(under):
        # `_xxx_cmd_name` --> `cmd-name`
        next_under: int = 0
        try:
            next_under = x.index(under, 1)
        except ValueError:
            pass
        if not next_under:
            x = x.lstrip(under)
        else:
            x = x[next_under + 1 :]

    x = x.strip("-").lower().replace("_", "-")

    if not x:
        raise ValueError("Expected at least one character")

    return x


def convert_to_flag_string(x: str) -> str:
    return "--" + convert_to_command_string(x)


def convert_to_short_flag_string(x: str) -> str:
    return "-" + convert_to_command_string(x)[0]


def make_dataclass_from_func(
    func: Callable[..., Any],
    base_classes: Optional[Tuple[Type[Dataclass], ...]] = None,
) -> Type[Dataclass]:

    if base_classes is None:
        base_classes = ()

    fields: List[Tuple[str, Type[Any], Field]] = []  # type: ignore

    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    for param in signature.parameters.values():
        annotation: Type[Any] = type_hints.get(param.name, param.annotation)
        default: Any = param.default
        default_value: Any

        field_metadata: Field  # type: ignore

        if isinstance(default, Field):
            if default.default_factory is not MISSING:
                default_value = default.default_factory()
            else:
                default_value = default.default
            field_metadata = default
        else:
            if default is inspect._empty:  # pylint: disable=protected-access
                default_value = MISSING
            else:
                default_value = default

            field_metadata = arg(default=default_value)

        fallback_type: Type[Any] = str

        # pylint: disable=protected-access
        if annotation is inspect._empty:
            if default_value is MISSING:
                annotation = fallback_type
            elif default_value is None:
                annotation = Optional[fallback_type]
            else:
                type_of_default: Type[Any] = type(default_value)
                if type_of_default in (str, int, float, bool):
                    annotation = type_of_default
                else:
                    raise TypeError(
                        f"Provide explicit type annotation for '{param.name}' "
                        f"(type  of default is '{type_of_default})'"
                    )

        fields.append((param.name, annotation, field_metadata))

    dc: Type[Dataclass] = make_dataclass(
        cls_name="Dataclass_" + func.__name__,
        bases=base_classes,
        fields=fields,
    )

    dc.__doc__ = func.__doc__

    return dc
