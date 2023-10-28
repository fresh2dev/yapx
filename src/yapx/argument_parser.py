import argparse
import collections.abc
import sys
from collections import defaultdict
from contextlib import suppress
from dataclasses import MISSING, Field, fields
from enum import Enum
from functools import partial
from inspect import signature
from types import GeneratorType
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pkg_resources import get_distribution

from .actions import (
    BooleanOptionalAction,
    CountAction,
    FeatureFlagAction,
    HelpAction,
    HelpAllAction,
    PrePostAction,
    SplitCsvDictAction,
    SplitCsvListAction,
    SplitCsvSetAction,
    SplitCsvTupleAction,
)
from .arg import (
    ARGPARSE_ARG_METADATA_KEY,
    ArgparseArg,
    _eval_type,
    convert_to_command_string,
    convert_to_flag_string,
    get_type_args,
    get_type_metadata,
    get_type_origin,
    make_dataclass_from_func,
)
from .command import Command, CommandMap, CommandOrCallable, CommandSequence, cmd
from .context import Context
from .exceptions import NoArgsModelError, raise_unsupported_type_error
from .namespace import Namespace
from .types import Dataclass, Literal, NoneType
from .utils import (
    SUPPORTED_SHELLS,
    RawTextHelpFormatter,
    ValidationError,
    add_tui_argument,
    add_tui_command,
    cast_type,
    completion_action,
    create_pydantic_model_from_dataclass,
    get_action_result,
    is_dataclass_type,
    is_pydantic_available,
    is_shtab_available,
    is_tui_available,
    try_isinstance,
    try_issubclass,
)

if sys.version_info >= (3, 9):
    from typing import _AnnotatedAlias
else:
    from typing_extensions import _AnnotatedAlias

if sys.version_info >= (3, 10):
    from types import UnionType  # pylint: disable=unused-import # noqa: F401

T = TypeVar("T")


class ArgumentParser(argparse.ArgumentParser):
    #  ROOT_FUNC_ATTR_NAME: str = "_root_func"
    #  ROOT_FUNC_ARGS_ATTR_NAME: str = "_root_func_args_model"
    CMD_PARSER_ATTR_NAME: str = "_command_parser"
    CMD_FUNC_ATTR_NAME: str = "_command_func_"
    CMD_FUNC_ARGS_ATTR_NAME: str = "_command_args_model_"

    def __init__(
        self,
        *args: Any,
        prog: Optional[str] = None,
        prog_version: Optional[str] = None,
        description: Optional[str] = None,
        help_flags: Optional[List[str]] = None,
        tui_flags: Optional[List[str]] = None,
        version_flags: Optional[List[str]] = None,
        completion_flags: Optional[List[str]] = None,
        formatter_class: Type[Any] = RawTextHelpFormatter,
        add_help: Optional[bool] = True,
        add_help_all: Optional[bool] = None,
        _parent_parser: Optional["ArgumentParser"] = None,
        **kwargs: Any,
    ):
        self._depth: int = 0
        self._parent_parser = _parent_parser

        if self._parent_parser:
            self._depth = self._parent_parser._depth + 1

            description = None
            formatter_class = self._parent_parser.formatter_class
            if help_flags is None:
                help_flags = self._parent_parser._help_flags
            if tui_flags is None:
                tui_flags = self._parent_parser._tui_flags

        super().__init__(
            *args,
            prog=prog,
            description=description,
            formatter_class=formatter_class,
            add_help=False,
            **kwargs,
        )

        self._subparsers_action: Optional[argparse._SubParsersAction] = None

        self._positionals.title = "required parameters"
        self._optionals.title = "optional parameters"

        helpful_arg_group = self.add_argument_group(
            "helpful parameters",
        ).add_mutually_exclusive_group()

        self._action_groups = [self._action_groups[-1], *self._action_groups[:-1]]

        if add_help is False:
            help_flags = []

        if add_help_all is None:
            add_help_all = add_help and self._parent_parser is None

        self._help_flags = help_flags
        self._tui_flags = tui_flags

        if help_flags is None:
            help_flags = ["--help", "-h"]

        if help_flags:
            if isinstance(help_flags, str):
                help_flags = [help_flags]

            helpful_arg_group.add_argument(
                *help_flags,
                action=HelpAction,
                help="Show this help message.",
            )
            if add_help_all:
                help_all_flags = [f"{x}-all" for x in help_flags if x.startswith("--")]
                helpful_arg_group.add_argument(
                    *help_all_flags,
                    action=HelpAllAction,
                    help="Show help for all commands.",
                )

        self.kv_separator = "="

        self._mutually_exclusive_args: Dict[
            Optional[Callable[..., Any]],
            Dict[str, List[Tuple[str, Optional[str]]]],
        ] = defaultdict(lambda: defaultdict(list))

        self._dest_type: Dict[str, Union[type, Callable[[str], Any]]] = {}

        if not self._parent_parser:
            if version_flags is None:
                version_flags = ["--version"]

            if version_flags:
                if isinstance(version_flags, str):
                    version_flags = [version_flags]

                if self.prog and not prog_version:
                    with suppress(Exception):
                        prog_version = get_distribution(self.prog).version

                if prog_version:
                    helpful_arg_group.add_argument(
                        *version_flags,
                        action="version",
                        version=f"%(prog)s {prog_version}",
                        help="Show the program version number.",
                    )

            if completion_flags is None:
                completion_flags = ["--print-shell-completion"]

            if is_shtab_available and completion_flags:
                if isinstance(completion_flags, str):
                    completion_flags = [completion_flags]

                helpful_arg_group.add_argument(
                    *completion_flags,
                    action=completion_action(),
                    default=argparse.SUPPRESS,
                    choices=SUPPORTED_SHELLS,
                    help="Print shell completion script.",
                )

        if tui_flags is None:
            tui_flags = ["--tui"]

        if is_tui_available and tui_flags:
            if isinstance(tui_flags, str):
                tui_flags = [tui_flags]

            tui_help: str = "Show Textual User Interface (TUI)."

            if len(tui_flags) == 1 and not tui_flags[0].startswith("-"):
                if not self._parent_parser:
                    self._get_or_add_subparsers()
                    add_tui_command(
                        parser=self,
                        command=tui_flags[0],
                        help=tui_help,
                    )
            else:
                tui_flags = [convert_to_flag_string(x) for x in tui_flags]
                add_tui_argument(
                    parser=helpful_arg_group,
                    parent_parser=self._get_parser_chain()[0],
                    option_strings=tui_flags,
                    help=tui_help,
                )

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(2, f"error: {message}\n")

    def print_help(
        self,
        file: Optional[IO[str]] = None,
        include_commands: bool = False,
    ) -> None:
        """Print CLI help.

        Args:
            file: ...
            include_commands: if True, also print help for each command.

        Examples:
            >>> import yapx
            >>> from dataclasses import dataclass
            ...
            >>> @dataclass
            ... class AddNums:
            ...     x: int
            ...     y: int
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(AddNums)
            ...
            >>> parser.print_help(include_commands=True)  #doctest: +SKIP
        """
        sep_char: str = "_"
        separator: str = (sep_char * 80) + "\n"
        print()
        print(separator)
        print(f"$ {self.prog}")
        print(separator)

        # print usage last
        usage = self.usage
        self.usage = argparse.SUPPRESS

        if not file:
            file = sys.stdout

        super().print_help(file)

        self.usage = usage
        self._print_message("\n", file)
        self.print_usage(file)

        if include_commands and self._subparsers:
            if self._subparsers_action is None:
                self._subparsers_action = self._find_subparsers_action(parser=self)

            for _choice, subparser in self._subparsers_action.choices.items():
                subparser.print_help(file, include_commands=include_commands)

    def add_arguments(
        self,
        args_model: Union[Callable[..., Any], Type[Dataclass], Dict[str, Field]],
    ) -> None:
        """Add arguments from the given function or dataframe.

        Args:
            args_model: a function or dataclass from which to derive arguments.

        Examples:
            >>> import yapx
            >>> from dataclasses import dataclass
            ...
            >>> @dataclass
            ... class AddNums:
            ...     x: int
            ...     y: int
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(AddNums)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            >>> (parsed.x, parsed.y)
            (1, 2)

            >>> import yapx
            ...
            >>> def add_nums(x: int, y: int):
            ...     return x + y
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(add_nums)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            >>> (parsed.x, parsed.y)
            (1, 2)
        """
        self._add_arguments(self, args_model)

    def add_command(
        self,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "ArgumentParser":
        """Create a new subcommand and add arguments from the given function or dataframe to it.

        Args:
            args_model: a function or dataclass from which to derive arguments.
            name: name of the command
            **kwargs: passed to `subparsers.add_parser(...)`

        Returns:
            the new subparser

        Examples:
            >>> import yapx
            >>> from dataclasses import dataclass
            ...
            >>> @dataclass
            ... class AddNums:
            ...     x: int
            ...     y: int
            ...
            >>> parser = yapx.ArgumentParser()
            >>> subparser_1 = parser.add_command(AddNums, name='add')
            >>> subparser_2 = parser.add_command(AddNums, name='subtract')
            ...
            >>> parsed = parser.parse_args(['add', '-x', '1', '-y', '2'])
            >>> (parsed.x, parsed.y)
            (1, 2)

            >>> import yapx
            ...
            >>> def add_nums(x: int, y: int):
            ...     return x + y
            ...
            >>> def subtract_nums(x: int, y: int):
            ...     return x - y
            ...
            >>> parser = yapx.ArgumentParser()
            >>> subparser_1 = parser.add_command(add_nums, name='add')
            >>> subparser_2 = parser.add_command(subtract_nums, name='subtract')
            ...
            >>> parsed = parser.parse_args(['subtract', '-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
        """
        # pylint: disable=protected-access
        subparsers: argparse._SubParsersAction = self._get_or_add_subparsers()

        if not name:
            name = convert_to_command_string(args_model.__name__)

        prog: Optional[str] = kwargs.pop("prog", None)
        if prog:
            prog += " " + name

        # subparsers must populate 'help' in order to # be included in shell-completion!
        # ref: https://github.com/iterative/shtab/issues/54#issuecomment-940516963
        description_txt: Optional[str] = self._get_description_from_docstring(
            args_model=args_model,
        )
        help_txt: Optional[str] = None
        if description_txt:
            help_txt = description_txt.splitlines()[0]

        parser: "ArgumentParser" = subparsers.add_parser(
            name,
            prog=prog,
            help=help_txt,
            **kwargs,
        )
        if description_txt:
            parser.description = description_txt

        if args_model:
            self._add_arguments(parser, args_model)

        return parser

    def _register_command(
        self,
        command: Union[str, Callable[..., Any], Command],
    ) -> "ArgumentParser":
        c: Command = (
            cmd(None, name=command) if isinstance(command, str) else cmd(command)
        )

        parser: "ArgumentParser" = self.add_command(
            args_model=c.function,
            name=c.name,
            **c.kwargs,
        )
        parser.set_defaults(
            **{
                self.CMD_PARSER_ATTR_NAME: parser,
                # pylint: disable=protected-access
                self.CMD_FUNC_ATTR_NAME + str(parser._depth): c.function,
            },
        )
        return parser

    @staticmethod
    def _is_attribute_inherited(dcls: Type[Dataclass], attr: str) -> bool:
        return attr not in dcls.__annotations__

    @classmethod
    def _add_arguments(
        cls,
        parser: "ArgumentParser",
        args_model: Union[Callable[..., Any], Type[Dataclass], Dict[str, Field]],
    ) -> List[ArgparseArg]:
        # Derived from: https://github.com/mivade/argparse_dataclass

        model: Type[Union[Dataclass, Dict[Any, Any]]]

        arg_fields: Tuple[Field, ...]

        if isinstance(args_model, dict):
            for k, v in args_model.items():
                v.name = k
            arg_fields = tuple(args_model.values())
            model = dict
        else:
            if is_dataclass_type(args_model):
                model = args_model
            else:
                model = make_dataclass_from_func(args_model)

            arg_fields = [
                f
                for f in fields(model)
                if not cls._is_attribute_inherited(dcls=model, attr=f.name)
            ]

        # pylint: disable=protected-access
        if parser.get_default(cls.CMD_FUNC_ARGS_ATTR_NAME + str(parser._depth)):
            err: str = "This parser already contains arguments from another dataclass."
            parser.error(err)

        parser.set_defaults(
            **{(parser.CMD_FUNC_ARGS_ATTR_NAME + str(parser._depth)): model},
        )

        parser_arg_groups: Dict[str, argparse._ArgumentGroup] = {
            x.title: x for x in parser._action_groups if x.title
        }

        novel_field_args: List[Tuple[Field, ArgparseArg]] = [
            (f, f.metadata.get(ARGPARSE_ARG_METADATA_KEY, ArgparseArg()))
            for f in arg_fields
        ]

        added_args: List[ArgparseArg] = []

        for fld, argparse_argument in novel_field_args:
            fld_type: Union[str, Type[Any]] = fld.type

            # basic support for handling deferred annotation evaluation,
            # when using `from __future__ import annotations`
            if try_isinstance(fld_type, str):
                fld_type = _eval_type(fld_type)
            assert not try_isinstance(fld_type, str)

            if (
                sys.version_info >= (3, 10) and try_isinstance(fld_type, UnionType)
            ) or get_type_origin(fld_type) is Union:
                fld_type = cls._extract_type_from_container(
                    fld_type,
                    is_union_type=True,
                )

            if try_isinstance(fld_type, _AnnotatedAlias):
                for x in get_type_metadata(fld_type):
                    if isinstance(x, Field) and ARGPARSE_ARG_METADATA_KEY in x.metadata:
                        argparse_argument = x.metadata[ARGPARSE_ARG_METADATA_KEY]
                        break

                fld_type = get_type_origin(fld_type)

                if (
                    sys.version_info >= (3, 10) and try_isinstance(fld_type, UnionType)
                ) or get_type_origin(fld_type) is Union:
                    fld_type = cls._extract_type_from_container(
                        fld_type,
                        is_union_type=True,
                    )

            default_metavar: str = "#" if fld_type in (int, float) else "value"

            argparse_argument.set_dest(fld.name)

            if argparse_argument.required:
                if fld.default is not MISSING:
                    argparse_argument.default = fld.default
                    argparse_argument.required = False
                elif fld.default_factory is not MISSING:
                    argparse_argument.default = fld.default_factory()
                    argparse_argument.required = False

            fld_type_origin: Optional[Type[Any]] = get_type_origin(fld_type)
            if fld_type_origin:
                if fld_type_origin and fld_type_origin is Literal:
                    argparse_argument.choices = list(get_type_args(fld_type))
                    if argparse_argument.choices:
                        fld_type = type(argparse_argument.choices[0])
                else:
                    # this is a type-container (List, Set, Tuple, Dict, ...)
                    if try_issubclass(fld_type_origin, collections.abc.Mapping):
                        fld_type = cls._extract_type_from_container(
                            fld_type,
                            assert_primitive=True,
                        )
                        default_metavar = f"key={default_metavar}"
                        if not argparse_argument.action:
                            argparse_argument.action = SplitCsvDictAction

                    elif (
                        try_issubclass(fld_type_origin, collections.abc.Iterable)
                        or fld_type_origin is set
                    ):
                        fld_type = cls._extract_type_from_container(
                            fld_type,
                            assert_primitive=True,
                        )
                        if not argparse_argument.action:
                            if fld_type_origin is set:
                                argparse_argument.action = SplitCsvSetAction
                            elif fld_type_origin is tuple:
                                argparse_argument.action = SplitCsvTupleAction
                            else:
                                argparse_argument.action = SplitCsvListAction

                    elif not is_pydantic_available:
                        raise_unsupported_type_error(fld_type)

                    if try_issubclass(fld_type, Enum):
                        argparse_argument.choices = [x.name for x in fld_type]

                    if isinstance(parser, cls):
                        # store desired types for casting later
                        # pylint: disable=protected-access
                        parser._dest_type[argparse_argument.dest] = partial(
                            cast_type,
                            fld_type,
                        )

                    unbounded_arg: bool = False
                    with suppress(ValueError, TypeError):
                        if int(argparse_argument.nargs) < 0:
                            unbounded_arg = True

                    if unbounded_arg or (
                        argparse_argument.pos and argparse_argument.nargs is None
                    ):
                        if argparse_argument.required:
                            argparse_argument.nargs = "+"
                        else:
                            argparse_argument.nargs = "*"

                    # type-containers must only contain strings
                    # until parsed by argparse.
                    fld_type = str

            elif try_issubclass(fld_type, Enum):
                argparse_argument.choices = list(fld_type)
            elif fld_type is None:
                fld_type = str

            argparse_argument.type = fld_type

            kwargs: Dict[str, Any] = argparse_argument.asdict()
            del kwargs["pos"]

            option_strings: Optional[List[str]] = kwargs.pop("option_strings")
            args: List[str] = (
                []
                if not option_strings
                else [x for x in option_strings if x.startswith("-")]
            )

            if not args:
                default_metavar = kwargs["dest"].upper()

            if not kwargs["metavar"] and not kwargs.get("choices"):
                kwargs["metavar"] = f"<{default_metavar}>"

            required: bool = kwargs["required"]

            if not args:
                # positional arg
                del kwargs["required"]
                if not required and kwargs.get("nargs") is None:
                    kwargs["nargs"] = "?"
                kwargs["type"] = partial(cast_type, fld_type)
            elif argparse_argument.type is bool:
                if isinstance(parser, cls):
                    # store desired types for casting later
                    # pylint: disable=protected-access
                    parser._dest_type[argparse_argument.dest] = partial(
                        cast_type,
                        fld_type,
                    )

                if not kwargs["action"]:
                    kwargs["action"] = BooleanOptionalAction
            elif argparse_argument.nargs != 0:
                kwargs["type"] = partial(cast_type, fld_type)
            elif fld_type is int:
                # 'int' args with nargs==0 are "counting" parameters (-vvv).
                kwargs["action"] = CountAction
            elif fld_type is str:
                # 'str' args with nargs==0 are "feature flag" parameters.
                kwargs["action"] = FeatureFlagAction

            # if given `default` cast it to the expected type.
            if (
                fld.default is not MISSING
                or fld.default_factory is not MISSING
                or kwargs.get("default") is not None
            ):
                if try_issubclass(kwargs.get("action"), PrePostAction):
                    kwargs["default"] = get_action_result(
                        action=kwargs["action"],
                        parser=parser,
                        dest=kwargs["dest"],
                        default=kwargs["default"],
                        option_strings=args,
                    )
                elif "type" in kwargs:
                    kwargs["default"] = kwargs["type"](kwargs["default"])

                # validate parsed default against 'choices'
                if kwargs.get("choices"):
                    d: Any = kwargs["default"]
                    if try_isinstance(d, str) or not try_isinstance(
                        d,
                        collections.abc.Iterable,
                    ):
                        d = [d]

                    for x in d:
                        if (
                            x not in kwargs["choices"]
                            and (
                                not try_isinstance(x, Enum)
                                or x.name not in kwargs["choices"]
                            )
                            # not if arg is 'Optional' and value is 'None'
                            and not (not kwargs.get("required") and x is None)
                        ):
                            err: str = (
                                f"Invalid value '{x}' for argument '{kwargs['dest']}';"
                                f" must be one of: {kwargs['choices']}"
                            )
                            parser.error(err)

            help_msg_parts: List[str] = []

            # pylint: disable=protected-access
            default: Any = kwargs.get("default")
            if (
                not required
                and default is not None
                and (not hasattr(default, "__len__") or len(default) > 1)
                and argparse_argument.type is not bool
            ):
                if isinstance(default, str):
                    help_msg_parts.append('Default: "%(default)s"')
                else:
                    help_msg_parts.append("Default: %(default)s")

            if kwargs.pop("exclusive", False):
                help_msg_parts.append("M.X.")

            if argparse_argument._env_var:
                help_msg_parts.append(f"Env: {argparse_argument._env_var}")

            if help_msg_parts:
                help_msg: str = f"| {', '.join(help_msg_parts)}"

                if kwargs.get("help"):
                    kwargs["help"] += "\n" + help_msg
                else:
                    kwargs["help"] = help_msg

            group: Optional[str] = kwargs.pop("group", None)

            if not group:
                group = "required parameters" if required else "optional parameters"

            arg_group: argparse._ArgumentGroup = parser_arg_groups.get(group, None)

            if not arg_group:
                arg_group = parser.add_argument_group(group)
                parser_arg_groups[group] = arg_group

            if argparse_argument.exclusive:
                if required:
                    err: str = (
                        "A mutually-exclusive parameter cannot be required:"
                        f" {argparse_argument.dest}"
                    )
                    parser.error(err)

                parser._mutually_exclusive_args[
                    parser._defaults.get(parser.CMD_FUNC_ATTR_NAME + str(parser._depth))
                ][group].append(
                    (
                        argparse_argument.dest,
                        args[-1] if args else None,
                    ),
                )

            action_class: Union[None, str, Callable[[...], Any]] = kwargs.get("action")
            if action_class is not None:
                if isinstance(kwargs.get("action"), str):
                    action_class = parser._registries["action"].get(
                        kwargs["action"],
                    )
                assert callable(action_class)
                if try_issubclass(action_class, argparse._StoreConstAction):
                    kwargs["nargs"] = 0
                    kwargs["required"] = False
                    if try_issubclass(
                        action_class,
                        (argparse._StoreTrueAction, argparse._StoreFalseAction),
                    ):
                        kwargs["type"] = bool
                        if kwargs["default"] is None:
                            kwargs["default"] = not try_issubclass(
                                action_class,
                                argparse._StoreTrueAction,
                            )
                action_class_kwargs: List[str] = [
                    *signature(action_class).parameters,
                    "action",
                ]
                for k in list(kwargs):
                    if k not in action_class_kwargs:
                        del kwargs[k]

            arg_group.add_argument(*args, **kwargs)

            added_args.append(argparse_argument)

        return added_args

    @classmethod
    def _extract_type_from_container(
        cls,
        type_container: Type[Any],
        assert_primitive: bool = False,
        is_union_type: bool = False,
    ) -> Type[Any]:
        type_container_origin: Any = (
            Union if is_union_type else get_type_origin(type_container)
        )

        if type_container_origin is None:
            raise TypeError(
                f"Given type is not a container: {type_container.__name__}",
            )

        type_container_args = get_type_args(type_container)

        results: List[Type[Any]] = [
            a for a in type_container_args if a is not ... and a is not NoneType
        ]

        type_container_subtype: type

        if (
            type_container_origin is Union
            or try_issubclass(type_container_origin, collections.abc.Sequence)
            or type_container_origin is set
        ):
            if len(results) != 1:
                results.clear()
            else:
                type_container_subtype = results[0]
        elif try_issubclass(type_container_origin, collections.abc.Mapping):
            if len(results) != 2:
                results.clear()
            else:
                key_type, value_type = results
                if key_type is not str:
                    raise TypeError("Dictionary keys must be type `str`")
                type_container_subtype = value_type
        elif not is_pydantic_available:
            raise_unsupported_type_error(type_container)

        if not results:
            raise TypeError("Too many types in container")

        type_container_subtype_origin = get_type_origin(type_container_subtype)

        if type_container_subtype_origin is Union:
            type_container_subtype = cls._extract_type_from_container(
                type_container_subtype,
            )
            type_container_subtype_origin = get_type_origin(type_container_subtype)

        if (
            assert_primitive
            and type_container_subtype_origin
            and not is_pydantic_available
        ):
            raise_unsupported_type_error(type_container_subtype_origin)

        return type_container_subtype

    def add_subparsers(self, **kwargs) -> argparse._SubParsersAction:
        default_kwargs: Dict[str, Any] = {
            "title": "commands",
            "metavar": "<COMMAND>",
        }
        for k, v in default_kwargs.items():
            kwargs[k] = kwargs.get(k, v)

        kwargs["parser_class"] = kwargs.get(
            "parser_class",
            lambda **k: type(self)(**k, _parent_parser=self),
        )

        self._subparsers_action = super().add_subparsers(**kwargs)

        return self._subparsers_action

    @classmethod
    def _find_subparsers_action(
        cls,
        parser: "ArgumentParser",
    ) -> Optional[argparse._SubParsersAction]:
        # pylint: disable=protected-access
        if not parser._subparsers_action:
            for a in parser._actions:
                if isinstance(a, argparse._SubParsersAction):
                    parser._subparsers_action = a
                    break
            else:
                return None

        return parser._subparsers_action

    def _get_or_add_subparsers(self) -> argparse._SubParsersAction:
        if self._subparsers_action is not None:
            return self._subparsers_action

        if self._subparsers:
            self._find_subparsers_action(parser=self)
            assert self._subparsers_action
            return self._subparsers_action

        return self.add_subparsers()

    def _get_parser_chain(self) -> List["ArgumentParser"]:
        parser_chain: List["ArgumentParser"] = []
        next_parser: Optional["ArgumentParser"] = self
        while next_parser is not None:
            parser_chain.append(next_parser)
            # pylint: disable=protected-access
            next_parser = next_parser._parent_parser
        parser_chain.reverse()
        return parser_chain

    def _post_parse_args(  # type: ignore[override]
        self,
        namespace: argparse.Namespace,
    ) -> Namespace:
        func_mx_arg_groups: Dict[
            str,
            List[Tuple[str, Optional[str]]],
        ] = self._mutually_exclusive_args.get(
            getattr(namespace, self.CMD_FUNC_ATTR_NAME + str(self._depth), None),
            {},
        )
        for g_args in func_mx_arg_groups.values():
            # argparse will always return a list when the argument type is `list | None`,
            # so we exclude both null and empty values
            # e.g.,: parser = argparse.ArgumentParser(); g = parser.add_mutually_exclusive_group(); g.add_argument("test", nargs="*", default=None); g.add_argument("--test2", default=None); parser.parse_args(["--test2", "aye"])
            mx_flags_found: List[str] = [
                flag if flag else dest
                for dest, flag in g_args
                if hasattr(namespace, dest)
                for attr_value in [getattr(namespace, dest)]
                if (
                    (
                        try_isinstance(attr_value, str)
                        or not try_isinstance(
                            attr_value,
                            collections.abc.Iterable,
                        )
                    )
                    and attr_value is not None
                )
                or (
                    not try_isinstance(attr_value, str)
                    and try_isinstance(
                        attr_value,
                        collections.abc.Iterable,
                    )
                    and attr_value
                )
            ]

            if len(mx_flags_found) > 1:
                err: str = "These arguments are mutually exclusive: " + ", ".join(
                    mx_flags_found,
                )
                self.error(err)

        return Namespace(**vars(namespace))

    def parse_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Namespace:
        try:
            parsed: argparse.Namespace = super().parse_args(
                args=args,
                namespace=namespace,
            )
            return self._post_parse_args(parsed)
        except ValueError as e:
            self.error(str(e))

    def parse_known_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[Namespace, List[str]]:
        parsed: argparse.Namespace
        unknown: List[str]
        try:
            parsed, unknown = super().parse_known_args(args=args, namespace=namespace)
            return self._post_parse_args(parsed), unknown
        except ValueError as e:
            self.error(str(e))

    def parse_known_args_to_model(
        self,
        args: Optional[Sequence[str]] = None,
        args_model: Optional[Type[Dataclass]] = None,
        skip_pydantic_validation: bool = False,
    ) -> Tuple[Dataclass, List[str]]:
        """Use parsed args to instantiate the given data model.

        Args:
            args: ...
            args_model: ...
            skip_pydantic_validation: ...

        Returns:
            ...

        Examples:
            >>> import yapx
            >>> from dataclasses import dataclass
            ...
            >>> @dataclass
            ... class AddNums:
            ...     x: int
            ...     y: int
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(AddNums)
            >>> parsed, unknown = parser.parse_known_args_to_model(['-x', '1', '-y', '2', '-z', '3'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> unknown
            ['-z', '3']

        """

        parsed_args: argparse.Namespace
        unknown_args: List[str]
        parsed_args, unknown_args = self.parse_known_args(args)

        return (
            self._parse_args_to_model(
                args=vars(parsed_args),
                args_model=args_model,
                skip_pydantic_validation=skip_pydantic_validation,
            ),
            unknown_args,
        )

    def parse_args_to_model(
        self,
        args: Optional[Sequence[str]] = None,
        args_model: Optional[Type[Dataclass]] = None,
        skip_pydantic_validation: bool = False,
    ) -> Dataclass:
        """Use parsed args to instantiate the given data model.

        Args:
            args: ...
            args_model: ...
            skip_pydantic_validation: ...

        Returns:
            ...

        Examples:
            >>> import yapx
            >>> from dataclasses import dataclass
            ...
            >>> @dataclass
            ... class AddNums:
            ...     x: int
            ...     y: int
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(AddNums)
            >>> parsed = parser.parse_args_to_model(['-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)

        """
        parsed_args: argparse.Namespace = self.parse_args(args)
        return self._parse_args_to_model(
            args=vars(parsed_args),
            args_model=args_model,
            skip_pydantic_validation=skip_pydantic_validation,
        )

    def _parse_args_to_model(
        self,
        args: Dict[str, Any],
        args_model: Optional[Type[Dataclass]] = None,
        skip_pydantic_validation: bool = False,
    ) -> Dataclass:
        if not args_model:
            args_model = args.get(self.CMD_FUNC_ARGS_ATTR_NAME + str(self._depth))
            if not args_model:
                raise NoArgsModelError

        args_union: Dict[str, Any] = self._union_args_with_model(
            args_dict=args,
            args_model=args_model,
        )

        if is_pydantic_available and not skip_pydantic_validation:
            # pylint: disable=not-callable
            try:
                args_union = vars(
                    create_pydantic_model_from_dataclass(args_model)(**args_union),
                )
            except ValidationError as e:
                err: str = "\n" + "\n".join(
                    f"Error parsing argument `{x['loc'][0]}`; {x['msg']}."
                    for x in e.errors()
                )
                self.error(err)

        return args_model(**args_union)

    @staticmethod
    def _union_args_with_model(
        args_dict: Dict[str, Any],
        args_model: Type[Dataclass],
    ) -> Dict[str, Any]:
        return {k: v for k, v in args_dict.items() if k in args_model.__annotations__}

    @classmethod
    def _get_description_from_docstring(
        cls,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
    ) -> Optional[str]:
        description_lines: Optional[List[str]] = None

        if args_model.__doc__ and not args_model.__doc__.startswith(
            args_model.__name__ + "(",
        ):
            description_lines = args_model.__doc__.strip().splitlines()

        if not description_lines:
            return None

        text_block_ends_at: int = 0
        for i, line in enumerate(description_lines):
            if not line.strip():
                text_block_ends_at = i
                break

        if text_block_ends_at > 0:
            description_lines = description_lines[:text_block_ends_at]

        return "\n".join(x.strip() for x in description_lines)

    @classmethod
    def _set_description_from_docstring(
        cls,
        parser: argparse.ArgumentParser,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
    ) -> None:
        description: Optional[str] = cls._get_description_from_docstring(args_model)

        if description:
            parser.description = description

    @classmethod
    def _run_func(
        cls,
        func: Callable[..., Any],
        context: Context,
        unknown_args: List[str],
        command_chain: List[Optional[Callable[..., Any]]],
        args_model: Optional[Type[Any]] = None,
    ) -> Any:
        context_arg_name: Optional[str] = None
        var_arg_name: Optional[str] = None
        var_kwarg_name: Optional[str] = None

        for p in signature(func).parameters.values():
            if p.kind is p.VAR_POSITIONAL:
                var_arg_name = p.name
            elif p.kind is p.VAR_KEYWORD:
                var_kwarg_name = p.name
            elif p.annotation is Context or Context in get_type_args(p.annotation):
                context_arg_name = p.name

        if (
            unknown_args
            and not var_arg_name
            and not var_kwarg_name
            and not any(
                p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                for x in command_chain
                if x is not None
                for p in signature(x).parameters.values()
            )
        ):
            context.parser.error(
                f"Unrecognized arguments: {' '.join(unknown_args)}",
            )

        if not args_model:
            args_model = make_dataclass_from_func(func)

        model_kwargs: Dict[str, Any] = vars(context.namespace)

        if var_arg_name or var_kwarg_name:
            if var_arg_name:
                model_kwargs[var_arg_name] = get_action_result(
                    action=SplitCsvTupleAction,
                    parser=context.subparser,
                    dest=var_arg_name,
                    default=unknown_args,
                    option_strings=[],
                )
            if var_kwarg_name:
                model_kwargs[var_kwarg_name] = get_action_result(
                    action=SplitCsvDictAction,
                    parser=context.subparser,
                    dest=var_kwarg_name,
                    default=unknown_args,
                    option_strings=[],
                )

        model_inst: Dataclass = (
            context.parser._parse_args_to_model(  # pylint: disable=protected-access
                args=model_kwargs,
                args_model=args_model,
            )
        )

        func_var_kwargs: Dict[str, Optional[Any]] = vars(model_inst)

        func_var_kwargs.update(func_var_kwargs.pop(var_kwarg_name, {}))

        if context_arg_name:
            func_var_kwargs[context_arg_name] = context

        func_var_args: List[str] = func_var_kwargs.pop(var_arg_name, [])

        return func(*func_var_args, **func_var_kwargs)

    def _build_subparsers(
        self,
        subcommands: Union[str, CommandOrCallable, CommandSequence, CommandMap],
    ) -> None:
        # pylint: disable=protected-access
        if isinstance(subcommands, dict):
            for parent, children in subcommands.items():
                subparser: "ArgumentParser" = (
                    self if parent is None else self._register_command(parent)
                )
                subparser._build_subparsers(children)
        else:
            if not isinstance(subcommands, Iterable):
                subcommands = [subcommands]

            for x in subcommands:
                self._register_command(x)

    @classmethod
    def _build_parser(
        cls,
        command: Optional[Callable[..., Any]] = None,
        subcommands: Union[
            None,
            str,
            CommandOrCallable,
            CommandSequence,
            CommandMap,
        ] = None,
        **kwargs: Any,
    ) -> "ArgumentParser":
        parser: "ArgumentParser" = cls(**kwargs)

        root_arg_model: Optional[Type[Dataclass]] = None

        if command:
            root_arg_model = make_dataclass_from_func(command)
            cls._set_description_from_docstring(
                parser=parser,
                args_model=root_arg_model,
            )
            parser.add_arguments(root_arg_model)

        parser.set_defaults(
            **{
                #  cls.ROOT_FUNC_ATTR_NAME: command,
                #  cls.ROOT_FUNC_ARGS_ATTR_NAME: root_arg_model,
                cls.CMD_PARSER_ATTR_NAME: parser,
                cls.CMD_FUNC_ATTR_NAME + str(parser._depth): command,
                cls.CMD_FUNC_ARGS_ATTR_NAME + str(parser._depth): root_arg_model,
            },
        )

        if subcommands:
            parser._build_subparsers(subcommands)

        return parser

    @classmethod
    def _run(
        cls,
        *parser_args: Any,
        args: Optional[List[str]] = None,
        default_args: Optional[List[str]] = None,
        **parser_kwargs: Any,
    ) -> Any:
        """Use given functions to construct an ArgumentParser,
        parse the args, invoke the appropriate command, and return any result.

        Args:
            *parser_args: ...
            args: ...
            default_args: ...
            **parser_kwargs: ...

        Returns:
            ...

        Examples:
            >>> import yapx
            ...
            >>> def print_nums(*args: int):
            ...     print('Args: ', *args)
            ...     return args
            ...
            >>> def find_evens(_context: yapx.Context):
            ...     return [x for x in _context.relay_value if x % 2 == 0]
            ...
            >>> def find_odds(_context: yapx.Context):
            ...     return [x for x in _context.relay_value if x % 2 != 0]
            ...
            >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
            >>> yapx.run(print_nums, [find_evens, find_odds], args=cli_args)
            Args:  1 2 3 4 5
            [1, 3, 5]
        """
        parser: ArgumentParser = cls._build_parser(*parser_args, **parser_kwargs)

        if args is None:
            args = sys.argv[1:]

        if not args and default_args:
            args = default_args

        known_args: Namespace
        unknown_args: List[str]
        known_args, unknown_args = parser.parse_known_args(args)

        relay_value: Any = None
        latest_return_value: Any = None

        teardown_generators: List[Generator] = []

        context: Context = Context(
            parser=parser,
            subparser=None,
            args=args,
            namespace=known_args,
            relay_value=relay_value,
        )

        cmd_parser: "ArgumentParser" = getattr(
            context.namespace,
            context.parser.CMD_PARSER_ATTR_NAME,
        )

        # pylint: disable=protected-access
        parser_chain: List["ArgumentParser"] = cmd_parser._get_parser_chain()

        parser_cmd_map: List[
            Tuple["ArgumentParser", Optional[Callable[..., Any]], Optional[Dataclass]]
        ] = [
            (
                parser,
                getattr(
                    known_args,
                    parser.CMD_FUNC_ATTR_NAME + str(parser._depth),
                ),
                getattr(
                    known_args,
                    parser.CMD_FUNC_ARGS_ATTR_NAME + str(parser._depth),
                ),
            )
            for parser in parser_chain
        ]

        command_chain: List[Optional[Callable[..., Any]]] = [
            x[1] for x in parser_cmd_map
        ]

        for cmd_parser, cmd_func, cmd_func_args_model in parser_cmd_map:
            if cmd_func is None:
                continue

            context = Context(**{**vars(context), "subparser": cmd_parser})

            latest_return_value = cls._run_func(
                func=cmd_func,
                context=context,
                args_model=cmd_func_args_model,
                unknown_args=unknown_args,
                command_chain=command_chain,
            )

            if not try_isinstance(latest_return_value, GeneratorType):
                relay_value = latest_return_value
            else:
                try:
                    relay_value = next(latest_return_value)
                    teardown_generators.append(latest_return_value)
                except StopIteration:
                    relay_value = latest_return_value

            context = Context(**{**vars(context), "relay_value": relay_value})

        for x in reversed(teardown_generators):
            with suppress(StopIteration):
                while True:
                    next(x)

        return relay_value
