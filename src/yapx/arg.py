import os
import sys
from argparse import Action
from contextlib import suppress
from dataclasses import MISSING, Field, dataclass, field, make_dataclass
from inspect import Parameter, _empty, signature
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import yapx  # pylint: disable=unused-import # noqa: F401

from .types import Dataclass

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=unused-import # noqa: F401
else:
    from typing_extensions import Literal  # pylint: disable=unused-import # noqa: F401

if sys.version_info >= (3, 9):
    from typing import Annotated  # pylint: disable=unused-import # noqa: F401
else:
    from typing_extensions import (  # pylint: disable=unused-import # noqa: F401
        Annotated,
    )


__all__ = ["arg"]


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
    _env_var: Union[None, str, Sequence[str]] = None

    def __post_init__(self) -> None:
        self.set_dest(self.dest)

    def set_dest(self, value: Optional[str]) -> None:
        self.dest = value

        if self.option_strings:
            if self.pos:
                self.option_strings = None
            elif isinstance(self.option_strings, str):
                self.option_strings = [self.option_strings]
        elif self.dest and not self.pos:
            self.option_strings = [convert_to_flag_string(self.dest)]

    def asdict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def arg(
    default: Optional[Any] = MISSING,
    env: Union[None, str, Sequence[str]] = None,
    pos: Optional[bool] = False,
    group: Optional[str] = None,
    exclusive: Optional[bool] = False,
    flags: Union[None, str, Sequence[str]] = None,
    # pylint: disable=redefined-builtin
    help: Optional[str] = None,
    metavar: Optional[str] = None,
    action: Union[None, str, Type[Action]] = None,
) -> Field:
    if env:
        if isinstance(env, str):
            env = [env]
        for e in env:
            value_from_env = os.getenv(e, None)
            if value_from_env:
                default = value_from_env
                break

            env_file = os.getenv(e + "_FILE", None)
            if env_file and os.path.exists(env_file):
                with open(env_file, encoding="utf8") as f:
                    value_from_file = f.read().strip()
                if value_from_file:
                    default = value_from_file
                    break

    metadata: Dict[str, ArgparseArg] = {
        ARGPARSE_ARG_METADATA_KEY: ArgparseArg(
            action=action,
            pos=pos,
            option_strings=flags,
            required=bool(default is MISSING),
            default=(
                None
                if default is MISSING
                else default() if callable(default) else default
            ),
            group=group,
            exclusive=exclusive,
            help=help,
            metavar=metavar,
            _env_var=env,
        ),
    }

    kwargs: Dict[str, Any] = {"metadata": metadata}

    default_param: str = "default_factory" if callable(default) else "default"
    kwargs[default_param] = default

    return field(**kwargs)


def convert_to_command_string(x: str) -> str:
    x = x.strip()

    x_prefix: str = "x_"
    under: str = "_"
    if x.startswith(x_prefix):
        x = x[len(x_prefix) :]
    elif x.startswith(under):
        # `_xxx_cmd_name` --> `cmd-name`
        next_under: int = 0
        with suppress(ValueError):
            next_under = x.index(under, 1)
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


def _eval_type(type_str: str) -> Type[Any]:
    if "[" in type_str:
        # None | list[str] --> None | List[str]
        type_str = "|".join(
            (y.capitalize() if "[" in y else y)
            for x in type_str.split("|")
            for y in [x.strip()]
        )
    if "|" in type_str:
        # None | str --> Union[None, str]
        type_str = f"Union[{type_str.replace('|', ',')}]"

    type_obj: Type[Any] = eval(type_str)  # pylint: disable=eval-used

    return type_obj


def make_dataclass_from_func(
    func: Callable[..., Any],
    base_classes: Optional[Tuple[Type[Dataclass], ...]] = None,
    include_private_params: bool = False,
) -> Type[Dataclass]:
    if base_classes is None:
        base_classes = ()

    fields: List[Tuple[str, Type[Any], Field]] = []

    func_signature = signature(func)
    type_hints: Dict[str, Any]
    try:
        include_extras: Dict[str, bool] = {}
        if sys.version_info >= (3, 9):
            include_extras["include_extras"] = True
        type_hints = get_type_hints(func, **include_extras)
    except (TypeError, NameError):
        # this can happen if deferred evaluation is used,
        # via `from __future__ import annotations`
        type_hints = {}

    param: Parameter
    for param in func_signature.parameters.values():
        if str(param).startswith("*") or (
            not include_private_params and param.name.startswith("_")
        ):
            continue

        annotation: Type[Any] = (
            _eval_type(param.annotation)
            if isinstance(param.annotation, str)
            else type_hints.get(param.name, param.annotation)
        )

        default: Any = param.default
        default_value: Any

        field_metadata: Field

        if isinstance(default, Field):
            if default.default_factory is not MISSING:
                default_value = default.default_factory()
            else:
                default_value = default.default
            field_metadata = default
        else:
            if default is _empty:  # pylint: disable=protected-access
                default_value = MISSING
            else:
                default_value = default

            field_metadata = arg(default=default_value)

        fallback_type: Type[Any] = str

        # pylint: disable=protected-access
        if annotation is _empty:
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
                        (
                            f"Provide explicit type annotation for '{param.name}' "
                            f"(type  of default is '{type_of_default})'"
                        ),
                    )

        fields.append((param.name, annotation, field_metadata))

    dc: Type[Dataclass] = make_dataclass(
        cls_name="Dataclass_" + func.__name__,
        bases=base_classes,
        fields=fields,
    )

    dc.__doc__ = func.__doc__

    return dc
