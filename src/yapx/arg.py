import argparse
import os
import sys
from dataclasses import MISSING, Field, dataclass, field, make_dataclass
from inspect import Parameter, signature
from pathlib import Path
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

from .context import Context
from .types import Dataclass

if sys.version_info >= (3, 8):
    from typing import Literal  # pylint: disable=unused-import # noqa: F401
else:
    from typing_extensions import Literal  # pylint: disable=unused-import # noqa: F401

if sys.version_info >= (3, 9):
    # pylint: disable=unused-import
    from typing import Annotated  # noqa: F401
    from typing import _AnnotatedAlias  # noqa: F401
else:
    # pylint: disable=unused-import
    from typing_extensions import Annotated  # noqa: F401
    from typing_extensions import _AnnotatedAlias  # noqa: F401


ARGPARSE_ARG_METADATA_KEY: str = "_argparse_argument"


def get_type_origin(t: Type[Any]) -> Optional[Type[Any]]:
    return getattr(t, "__origin__", None)


def get_type_args(t: Type[Any]) -> Tuple[Type[Any], ...]:
    return getattr(t, "__args__", ())


def get_type_metadata(t: Type[Any]) -> Tuple[Type[Any], ...]:
    return getattr(t, "__metadata__", ())


class DummyArgAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        help=None,  # pylint: disable=redefined-builtin
        metavar=None,
        **_kwargs,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            help=help,
            metavar=metavar,
            nargs=0,
            default=argparse.SUPPRESS,
            required=False,
            const=None,
            type=None,
            choices=None,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        pass


@dataclass
class ArgparseArg:
    dest: Optional[str] = None
    option_strings: Optional[Sequence[str]] = None
    type: Union[None, Type[Any], Callable[[str], Any]] = None
    action: Union[None, str, Type[argparse.Action]] = None
    required: bool = True
    group: Optional[str] = None
    exclusive: Optional[bool] = False
    nargs: Optional[Union[int, str]] = None
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

        if not self.option_strings and self.dest and not self.pos:
            self.option_strings = [self.dest]

        if not self.option_strings:
            self.option_strings = None
        else:
            if isinstance(self.option_strings, str):
                self.option_strings = [self.option_strings]

            self.option_strings = [
                convert_to_flag_string(x) for x in self.option_strings
            ]

            if self.pos:
                err: str = f"Positional arguments cannot have flags: {' '.join(self.option_strings)}"
                raise ValueError(err)

    def asdict(self) -> Dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def arg(
    *flags: str,
    default: Optional[Any] = MISSING,
    env: Union[None, str, Sequence[str]] = None,
    pos: Optional[bool] = False,
    group: Optional[str] = None,
    exclusive: Optional[bool] = False,
    help: Optional[str] = None,  # pylint: disable=redefined-builtin
    metavar: Optional[str] = None,
    nargs: Optional[Union[int, str]] = None,
    action: Union[None, str, Type[argparse.Action]] = None,
) -> Field:
    """Provides an interface to modify argument options.

    Args:
        *flags: one or more flags to use for the argument.
        default: default value for the argument. Argument is required if no default is given.
        env: list of environment variables that will provide the argument value.
        pos: if True, argument is positional (no flags).
        group: group for the argument.
        exclusive: if True, this arg cannot be specified along with another exclusive arg in the same group.
        help: help text / description
        metavar: variable name printed in help text.
        nargs: the number of values this argument accepts.
        action: custom action for this argument.

    Returns:
        ...

    Examples:
        >>> import yapx
        >>> from yapx.types import Annotated
        ...
        >>> def say_hello(
        ...     value = yapx.arg(default='World')
        ... ):
        ...     print(f"Hello {value}")
        ...
        >>> yapx.run(say_hello, args=[])
        Hello World

        >>> import yapx
        >>> from yapx.types import Annotated
        ...
        >>> def say_hello(
        ...     value: Annotated[str, yapx.arg(default='World')]
        ... ):
        ...     print(f"Hello {value}")
        ...
        >>> yapx.run(say_hello, args=[])
        Hello World
    """
    if env:
        if isinstance(env, str):
            env = [env]
        for e in env:
            value_from_env = os.getenv(e, None)
            if value_from_env:
                default = value_from_env
                break

            env_file = os.getenv(e + "_FILE", None)
            if env_file:
                env_file_path: Path = Path(env_file)
                if env_file_path.exists():
                    value_from_file = env_file_path.read_text(encoding="utf-8").strip()
                    if value_from_file:
                        default = value_from_file
                        break

    all_flags: List[str] = []
    for x in flags:
        if isinstance(x, str):
            all_flags.append(x)
        else:
            all_flags.extend(x)

    metadata: Dict[str, ArgparseArg] = {
        ARGPARSE_ARG_METADATA_KEY: ArgparseArg(
            action=action,
            pos=pos,
            option_strings=flags,
            required=bool(default is MISSING),
            default=(
                None
                if default is MISSING
                else default()
                if callable(default)
                else default
            ),
            group=group,
            exclusive=exclusive,
            help=help,
            metavar=metavar,
            nargs=nargs,
            _env_var=env,
        ),
    }

    kwargs: Dict[str, Any] = {"metadata": metadata}

    default_param: str = "default_factory" if callable(default) else "default"
    kwargs[default_param] = default

    return field(**kwargs)


def _convert_to_command_strings(*args) -> List[str]:
    x_split: List[str] = [y for x in args for y in [x.strip(" _-")] if y]

    cmd_strings: List[str] = []

    for i in range(len(x_split)):
        x = x_split[i].replace(" ", "-").replace("_", "-")

        if not x:
            continue

        if not x.islower() and not x.isupper() and "-" not in x:
            # camelCase -> camel-case
            x = "".join(
                ["-" + c if c.isupper() and i > 0 else c for i, c in enumerate(x)],
            )

        if len(x) > 1:
            # avoid lower-casing single-letter commands/flags.
            x = x.lower()

        cmd_strings.append(x)

    if not cmd_strings:
        err: str = "Expected at least one valid character"
        raise ValueError(err)

    return cmd_strings


def convert_to_command_string(text: str) -> str:
    cmd_strings: List[str] = _convert_to_command_strings(text)
    assert len(cmd_strings) == 1
    return cmd_strings[0]


def convert_to_flag_string(text: str) -> str:
    return "/".join(
        f"--{x}" if len(x) > 1 else f"-{x}"
        for x in _convert_to_command_strings(*text.split("/"))
    )


def convert_to_short_flag_string(text: str) -> str:
    return "/".join(f"-{x[0]}" for x in _convert_to_command_strings(*text.split("/")))


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
) -> Type[Dataclass]:
    if base_classes is None:
        base_classes = ()

    fields: List[Tuple[str, Type[Any], Field]] = []

    func_signature = signature(func)
    type_hints: Dict[str, Any]
    try:
        if sys.version_info >= (3, 9):
            # pylint: disable=unexpected-keyword-arg
            type_hints = get_type_hints(func, include_extras=True)
        else:
            type_hints = get_type_hints(func)
    except (TypeError, NameError):
        # this can happen if deferred evaluation is used,
        # via `from __future__ import annotations`
        type_hints = {}

    params: List[Parameter] = list(func_signature.parameters.values())

    if params and params[0].kind is params[0].VAR_POSITIONAL:
        params = [*params[1:], params[0]]

    for param in params:
        annotation: Type[Any] = (
            _eval_type(param.annotation)
            if isinstance(param.annotation, str)
            else type_hints.get(param.name, param.annotation)
        )

        default: Any = param.default
        default_value: Any

        if annotation is Context or (
            param.name.startswith("_")
            and not isinstance(default, Field)
            and (
                not isinstance(annotation, _AnnotatedAlias)
                or not any(isinstance(x, Field) for x in get_type_metadata(annotation))
            )
        ):
            continue

        field_metadata: Field

        fallback_type: Type[Any] = str

        if param.kind is param.VAR_POSITIONAL:
            assert not isinstance(annotation, _AnnotatedAlias)
            assert param.default is param.empty
            field_metadata = arg(
                default=None,
                pos=True,
                nargs=0,
                action=DummyArgAction,
                metavar="<...>",
                help="Any extra command-line values.",
            )
            default_value = MISSING
            if annotation is param.empty:
                annotation = fallback_type
            annotation = Optional[Sequence[annotation]]
        elif param.kind is param.VAR_KEYWORD:
            assert not isinstance(annotation, _AnnotatedAlias)
            assert param.default is param.empty
            field_metadata = arg(
                default=None,
                pos=True,
                nargs=0,
                action=DummyArgAction,
                metavar="<x=y> ...",
                help="Any extra command-line key-value pairs.",
            )
            default_value = MISSING
            if annotation is param.empty:
                annotation = fallback_type
            annotation = Optional[Dict[str, Optional[annotation]]]
        elif isinstance(default, Field):
            if default.default_factory is not MISSING:
                default_value = default.default_factory()
            else:
                default_value = default.default
            field_metadata = default
        else:
            default_value = MISSING if default is param.empty else default
            field_metadata = arg(default=default_value)

        if annotation is param.empty:
            if default_value is MISSING:
                annotation = fallback_type
            elif default_value is None:
                annotation = Optional[fallback_type]
            else:
                type_of_default: Type[Any] = type(default_value)
                if type_of_default in (str, int, float, bool):
                    annotation = type_of_default
                else:
                    err: str = (
                        f"Provide explicit type annotation for '{param.name}' "
                        f"(type of default is '{type_of_default})'"
                    )
                    raise TypeError(err)

        fields.append((param.name, annotation, field_metadata))

    dc: Type[Dataclass] = make_dataclass(
        cls_name="Dataclass_" + func.__name__,
        bases=base_classes,
        fields=fields,
    )

    dc.__doc__ = func.__doc__

    return dc


def counting_arg(
    *args,
    **kwargs,
) -> Field:
    """Designates this argument as a counting argument.

    `yapx.counting_arg(...)` is equal to `yapx.arg(nargs=0, ...)`

    Must be used with a parameter annotated with type `int`.

    Args:
        *args: passed to arg(...)
        **kwargs: passed to arg(...)

    Returns:
        ...

    Examples:
        >>> import yapx
        >>> from yapx.types import Annotated
        >>> from typing import List
        ...
        >>> def say_hello(
        ...     verbosity: Annotated[int, yapx.feature_arg("v")]
        ... ):
        ...     print("verbosity:", verbosity)
        ...
        >>> yapx.run(say_hello, args=["-vvvvv"])
        verbosity: 5
    """
    kwargs["nargs"] = 0
    return arg(*args, **kwargs)


def feature_arg(
    *args,
    **kwargs,
) -> Field:
    """Designates this argument as a feature-flag argument.

    `yapx.feature_arg(...)` is equal to `yapx.arg(nargs=0, ...)`

    Must be used with a parameter annotated with type `str`.

    Args:
        *args: passed to arg(...)
        **kwargs: passed to arg(...)

    Returns:
        ...

    Examples:
        >>> import yapx
        >>> from yapx.types import Annotated
        >>> from typing import List
        ...
        >>> def say_hello(
        ...     value: Annotated[str, yapx.feature_arg("dev", "test", "prod")]
        ... ):
        ...     print(value)
        ...
        >>> yapx.run(say_hello, args=["--prod"])
        prod
    """
    kwargs["nargs"] = 0
    return arg(*args, **kwargs)


def unbounded_arg(
    *args,
    **kwargs,
) -> Field:
    """Designates this argument as an unbounded, multi-value argument.

    `yapx.unbounded_arg(...)` is equal to `yapx.arg(nargs=-1, ...)`

    Must be used with a parameter annotated with a sequence type:
    `Sequence[...]`, `List[...]`, `Set[...]`, `Tuple[..., ...]`, or `Dict[str, ...]`

    Args:
        *args: passed to arg(...)
        **kwargs: passed to arg(...)

    Returns:
        ...

    Examples:
        >>> import yapx
        >>> from yapx.types import Annotated
        >>> from typing import List
        ...
        >>> def say_hello(
        ...     values: Annotated[List[int], yapx.unbounded_arg()]
        ... ):
        ...     print(values)
        ...
        >>> yapx.run(say_hello, args=["--values", "1", "2", "3"])
        [1, 2, 3]
    """
    kwargs["nargs"] = -1
    return arg(*args, **kwargs)
