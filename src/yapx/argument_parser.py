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
    make_dataclass_from_func,
)
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
    ROOT_FUNC_ATTR_NAME: str = "_root_func"
    ROOT_FUNC_ARGS_ATTR_NAME: str = "_root_func_args_model"
    CMD_ATTR_NAME: str = "_command"
    CMD_FUNC_ATTR_NAME: str = "_command_func"
    CMD_FUNC_ARGS_ATTR_NAME: str = "_command_func_args_model"

    def __init__(
        self,
        *args: Any,
        prog: Optional[str] = None,
        prog_version: Optional[str] = None,
        description: Optional[str] = None,
        help_flags: Optional[List[str]] = None,
        version_flags: Optional[List[str]] = None,
        tui_flags: Optional[List[str]] = None,
        completion_flags: Optional[List[str]] = None,
        formatter_class: Type[Any] = RawTextHelpFormatter,
        _is_subparser: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            prog=prog,
            description=None if _is_subparser else description,
            add_help=False,
            formatter_class=formatter_class,
            **kwargs,
        )

        self._subparsers_action: Optional[argparse._SubParsersAction] = None

        # self._positionals.title = "commands"
        self._optionals.title = "helpful parameters"

        if help_flags is None:
            help_flags = ["-h", "--help"]

        if help_flags:
            if isinstance(help_flags, str):
                help_flags = [help_flags]
            self.add_argument(
                *[x for x in help_flags if x],
                action=HelpAction,
                help="Show this help message.",
            )

        self.kv_separator = "="

        self._mutually_exclusive_args: Dict[
            Optional[Callable[..., Any]],
            Dict[str, List[Tuple[str, Optional[str]]]],
        ] = defaultdict(lambda: defaultdict(list))

        self._dest_type: Dict[str, Union[type, Callable[[str], Any]]] = {}

        if not _is_subparser:
            if help_flags:
                help_all_flags = [f"{x}-all" for x in help_flags if x.startswith("--")]
                self.add_argument(
                    *help_all_flags,
                    action=HelpAllAction,
                    help="Show help for all commands.",
                )

            if version_flags is None:
                version_flags = ["--version"]

            if version_flags:
                if isinstance(version_flags, str):
                    version_flags = [version_flags]

                if self.prog and not prog_version:
                    with suppress(Exception):
                        prog_version = get_distribution(self.prog).version

                if prog_version:
                    self.add_argument(
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

                self.add_argument(
                    *completion_flags,
                    action=completion_action(),
                    default=argparse.SUPPRESS,
                    choices=SUPPORTED_SHELLS,
                    help="Print shell completion script.",
                )

            if tui_flags is None:
                tui_flags = ["--tui"]

            if tui_flags and is_tui_available:
                if isinstance(tui_flags, str):
                    tui_flags = [tui_flags]

                if len(tui_flags) == 1 and not tui_flags[0].startswith("-"):
                    self._get_or_add_subparsers()
                    add_tui_command(
                        parser=self,
                        option_strings=tui_flags[0],
                        help="Show Textual User Interface (TUI).",
                    )
                else:
                    add_tui_argument(
                        parser=self,
                        option_strings=tui_flags,
                        help="Show Textual User Interface (TUI).",
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

        # don't include usage in help text.
        usage = self.usage
        self.usage = argparse.SUPPRESS

        super().print_help(file)

        if include_commands and self._subparsers:
            if self._subparsers_action is None:
                self._subparsers_action = self._find_subparsers_action()

            for _choice, subparser in self._subparsers_action.choices.items():
                subparser.print_help(file, include_commands=include_commands)

        self.usage = usage

    def add_arguments(
        self,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
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
            >>> parser.set_defaults(_command_func=lambda x, y: x+y)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._command_func_args_model(x=parsed.x, y=parsed.y)
            AddNums(x=1, y=2)
            >>> parsed._command_func(x=parsed.x, y=parsed.y)
            3

            >>> import yapx
            ...
            >>> def add_nums(x: int, y: int):
            ...     return x + y
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(add_nums)
            >>> parser.set_defaults(_command_func=add_nums)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._command_func_args_model(x=parsed.x, y=parsed.y)
            Dataclass_add_nums(x=1, y=2)
            >>> parsed._command_func(x=parsed.x, y=parsed.y)
            3
        """
        self._add_arguments(self, args_model)

    def add_command(
        self,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> argparse.ArgumentParser:
        """Create a new subcommand and add arguments from the given function or dataframe to it.

        Args:
            args_model: a function or dataclass from which to derive arguments.
            name: name of the command

        Returns:
            the new argparse subparser

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
            >>> subparser_1.set_defaults(_command_func=lambda x, y: x+y)
            >>> subparser_2 = parser.add_command(AddNums, name='subtract')
            >>> subparser_2.set_defaults(_command_func=lambda x, y: x-y)
            ...
            >>> parsed = parser.parse_args(['add', '-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._command_func_args_model(x=parsed.x, y=parsed.y)
            AddNums(x=1, y=2)
            >>> parsed._command_func(x=parsed.x, y=parsed.y)
            3

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
            >>> subparser_1.set_defaults(_command_func=add_nums)
            >>> subparser_2 = parser.add_command(subtract_nums, name='subtract')
            >>> subparser_2.set_defaults(_command_func=subtract_nums)
            ...
            >>> parsed = parser.parse_args(['subtract', '-x', '1', '-y', '2'])
            ...
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._command_func_args_model(x=parsed.x, y=parsed.y)
            Dataclass_subtract_nums(x=1, y=2)
            >>> parsed._command_func(x=parsed.x, y=parsed.y)
            -1
        """
        subparsers: argparse.Action = self._get_or_add_subparsers()

        if not name:
            name = convert_to_command_string(args_model.__name__)

        # pylint: disable=protected-access
        assert isinstance(subparsers, argparse._SubParsersAction)
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

        parser = subparsers.add_parser(
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
        func: Callable[..., Any],
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        name = convert_to_command_string(name if name else func.__name__)
        parser: argparse.ArgumentParser = self.add_command(
            func,
            name=name,
            **kwargs,
        )
        parser.set_defaults(**{self.CMD_FUNC_ATTR_NAME: func})

    @staticmethod
    def _is_attribute_inherited(dcls: Type[Dataclass], attr: str) -> bool:
        return attr not in dcls.__annotations__

    @classmethod
    def _add_arguments(
        cls,
        parser: argparse.ArgumentParser,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
    ) -> List[ArgparseArg]:
        # Derived from: https://github.com/mivade/argparse_dataclass

        model: Type[Dataclass]

        if is_dataclass_type(args_model):
            model = args_model
        else:
            model = make_dataclass_from_func(args_model)

        if parser.get_default(cls.CMD_FUNC_ARGS_ATTR_NAME):
            err: str = "This parser already contains arguments from another dataclass."
            parser.error(err)

        parser.set_defaults(**{cls.CMD_FUNC_ARGS_ATTR_NAME: model})

        parser_arg_groups: Dict[str, argparse._ArgumentGroup] = {}

        novel_fields: List[Field] = list(
            filter(
                lambda f: not cls._is_attribute_inherited(dcls=model, attr=f.name),
                fields(model),
            ),
        )

        novel_field_args: List[Tuple[Field, ArgparseArg]] = [
            (x, x.metadata.get(ARGPARSE_ARG_METADATA_KEY, ArgparseArg()))
            for x in novel_fields
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
            ) or cls._get_type_origin(fld_type) is Union:
                fld_type = cls._extract_type_from_container(
                    fld_type,
                    is_union_type=True,
                )

            if try_isinstance(fld_type, _AnnotatedAlias):
                for x in cls._get_type_metadata(fld_type):
                    if isinstance(x, Field) and ARGPARSE_ARG_METADATA_KEY in x.metadata:
                        argparse_argument = x.metadata[ARGPARSE_ARG_METADATA_KEY]
                        break

                fld_type = cls._get_type_origin(fld_type)

                if (
                    sys.version_info >= (3, 10) and try_isinstance(fld_type, UnionType)
                ) or cls._get_type_origin(fld_type) is Union:
                    fld_type = cls._extract_type_from_container(
                        fld_type,
                        is_union_type=True,
                    )

            help_type: str = (
                fld_type.__name__
                if try_isinstance(fld_type, type)
                else str(fld_type).rsplit(".", maxsplit=1)[-1]
            )

            argparse_argument.set_dest(fld.name)

            if argparse_argument.required:
                if fld.default is not MISSING:
                    argparse_argument.default = fld.default
                    argparse_argument.required = False
                elif fld.default_factory is not MISSING:
                    argparse_argument.default = fld.default_factory()
                    argparse_argument.required = False

            fld_type_origin: Optional[Type[Any]] = cls._get_type_origin(fld_type)
            if fld_type_origin:
                if fld_type_origin and fld_type_origin is Literal:
                    argparse_argument.choices = list(cls._get_type_args(fld_type))
                    if argparse_argument.choices:
                        fld_type = type(argparse_argument.choices[0])
                else:
                    # this is a type-container (List, Set, Tuple, Dict, ...)
                    if try_issubclass(fld_type_origin, collections.abc.Mapping):
                        fld_type = cls._extract_type_from_container(
                            fld_type,
                            assert_primitive=True,
                        )
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

                    # type-containers must only contain strings
                    # until parsed by argparse.
                    fld_type = str

            elif try_issubclass(fld_type, Enum):
                argparse_argument.choices = list(fld_type)

            argparse_argument.type = fld_type

            kwargs: Dict[str, Any] = argparse_argument.asdict()
            del kwargs["pos"]

            option_strings: Optional[List[str]] = kwargs.pop("option_strings")
            args: List[str] = (
                []
                if not option_strings
                else [x for x in option_strings if x.startswith("-")]
            )

            if not kwargs["metavar"] and not kwargs.get("choices"):
                kwargs["metavar"] = (
                    f"<{kwargs['dest'].upper()}>" if not args else "<value>"
                )

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

                for k in "type", "nargs", "const", "choices", "metavar":
                    kwargs.pop(k, None)
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
                if kwargs.get("action") and try_issubclass(
                    kwargs["action"],
                    (argparse.Action, BooleanOptionalAction),
                ):
                    # https://stackoverflow.com/a/24448351
                    dummy_namespace: object = type("", (), {})()
                    kwargs["action"](option_strings=args, dest=kwargs["dest"])(
                        parser=parser,
                        namespace=dummy_namespace,
                        values=kwargs["default"],
                    )
                    kwargs["default"] = getattr(dummy_namespace, kwargs["dest"])
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

            help_msg_parts: List[str] = [f"Type: {help_type}"]

            # pylint: disable=protected-access
            if not required:
                if isinstance(kwargs.get("default"), str):
                    help_msg_parts.append('Default: "%(default)s"')
                else:
                    help_msg_parts.append("Default: %(default)s")

            if kwargs.pop("exclusive", False):
                help_msg_parts.append("M.X.")

            if argparse_argument._env_var:
                help_msg_parts.append(f"Env: {argparse_argument._env_var}")

            help_msg: str = f"> {', '.join(help_msg_parts)}"

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
                    parser._defaults.get(cls.CMD_FUNC_ATTR_NAME)
                ][group].append(
                    (
                        argparse_argument.dest,
                        args[-1] if args else None,
                    ),
                )

            arg_group.add_argument(*args, **kwargs)

            added_args.append(argparse_argument)

        return added_args

    @staticmethod
    def _get_type_origin(t: Type[Any]) -> Optional[Type[Any]]:
        return getattr(t, "__origin__", None)

    @staticmethod
    def _get_type_args(t: Type[Any]) -> Tuple[Type[Any], ...]:
        return getattr(t, "__args__", ())

    @staticmethod
    def _get_type_metadata(t: Type[Any]) -> Tuple[Type[Any], ...]:
        return getattr(t, "__metadata__", ())

    @classmethod
    def _extract_type_from_container(
        cls,
        type_container: Type[Any],
        assert_primitive: bool = False,
        is_union_type: bool = False,
    ) -> Type[Any]:
        type_container_origin: Any = (
            Union if is_union_type else cls._get_type_origin(type_container)
        )

        if type_container_origin is None:
            raise TypeError(
                f"Given type is not a container: {type_container.__name__}",
            )

        type_container_args = cls._get_type_args(type_container)

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

        type_container_subtype_origin = cls._get_type_origin(type_container_subtype)

        if type_container_subtype_origin is Union:
            type_container_subtype = cls._extract_type_from_container(
                type_container_subtype,
            )
            type_container_subtype_origin = cls._get_type_origin(type_container_subtype)

        if (
            assert_primitive
            and type_container_subtype_origin
            and not is_pydantic_available
        ):
            raise_unsupported_type_error(type_container_subtype_origin)

        return type_container_subtype

    def add_subparsers(self, **kwargs) -> argparse._SubParsersAction:
        kwargs["title"] = kwargs.get("title", "commands")
        kwargs["metavar"] = kwargs.get("metavar", "<COMMAND>")
        kwargs["dest"] = kwargs.get("dest", self.CMD_ATTR_NAME)
        kwargs["parser_class"] = kwargs.get(
            "parser_class",
            lambda **k: type(self)(
                **k,
                _is_subparser=True,
                formatter_class=self.formatter_class,
            ),
        )

        self._subparsers_action = super().add_subparsers(**kwargs)

        return self._subparsers_action

    def _find_subparsers_action(self) -> Optional[argparse._SubParsersAction]:
        for a in self._actions:
            if isinstance(
                a,
                argparse._SubParsersAction,  # pylint: disable=protected-access
            ):
                self._subparsers_action = a
                return self._subparsers_action
        return None

    def _get_or_add_subparsers(self) -> argparse._SubParsersAction:
        if self._subparsers_action is not None:
            return self._subparsers_action

        if self._subparsers:
            self._find_subparsers_action()
            assert self._subparsers_action
            return self._subparsers_action

        return self.add_subparsers()

    def _post_parse_args(  # type: ignore[override]
        self,
        namespace: argparse.Namespace,
    ) -> Namespace:
        func_mx_arg_groups: Dict[
            str,
            List[Tuple[str, Optional[str]]],
        ] = self._mutually_exclusive_args.get(
            getattr(namespace, self.CMD_FUNC_ATTR_NAME, None),
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
            args:
            args_model:
            skip_pydantic_validation:

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
                args=parsed_args,
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
            args:
            args_model:
            skip_pydantic_validation:

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
            args=parsed_args,
            args_model=args_model,
            skip_pydantic_validation=skip_pydantic_validation,
        )

    def _parse_args_to_model(
        self,
        args: argparse.Namespace,
        args_model: Optional[Type[Dataclass]] = None,
        skip_pydantic_validation: bool = False,
    ) -> Dataclass:
        parsed_args: Dict[str, Any] = vars(args)

        if not args_model:
            args_model = parsed_args.get(self.CMD_FUNC_ARGS_ATTR_NAME)
            if not args_model:
                raise NoArgsModelError

        args_union: Dict[str, Any] = self._union_args_with_model(
            args_dict=parsed_args,
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
        description: str = cls._get_description_from_docstring(args_model)

        if description:
            parser.description = description

    @classmethod
    def _run_func(
        cls,
        parser: "ArgumentParser",
        func: Callable[..., Any],
        args: List[str],
        args_model: Optional[Type[Any]] = None,
        linked_func: Optional[Callable[..., Any]] = None,
        relay_value: Optional[Any] = None,
    ) -> Any:
        func_args: List[str] = []
        func_kwargs: Dict[str, Optional[Any]] = {}

        extra_args_ok: bool = False

        accepts_args: bool = False
        accepts_extra_args: bool = False

        accepts_kwargs: bool = False
        accepts_extra_kwargs: bool = False

        for p in signature(func).parameters.values():
            if str(p).startswith("**"):
                accepts_kwargs = True
            elif str(p).startswith("*"):
                accepts_args = True
            elif p.name == "_extra_args":
                accepts_extra_args = True
            elif p.name == "_extra_kwargs":
                accepts_extra_kwargs = True
            elif p.name == "_all_args":
                func_kwargs["_all_args"] = args
            elif p.name == "_relay_value":
                func_kwargs["_relay_value"] = relay_value
            elif p.name == "_called_from_cli":
                func_kwargs["_called_from_cli"] = True

        extra_args_ok = (
            accepts_args or accepts_kwargs or accepts_extra_args or accepts_extra_kwargs
        )

        if not extra_args_ok and linked_func:
            extra_args_ok = any(
                str(p).startswith("*") or p.name in ("_extra_args", "_extra_kwargs")
                for p in signature(linked_func).parameters.values()
            )

        if not args_model:
            args_model = make_dataclass_from_func(func)

        model_inst: Dataclass
        if extra_args_ok:
            unknown_args: List[str]
            model_inst, unknown_args = parser.parse_known_args_to_model(
                args=args,
                args_model=args_model,
            )

            if accepts_kwargs or accepts_extra_kwargs:
                unknown_name_map: Dict[str, str] = {}
                unknown_parser: ArgumentParser = cls()
                for x in unknown_args:
                    if x.startswith("-"):
                        x_flag: str = x.split("=", maxsplit=1)[0]
                        x_flag_bare = x_flag.lstrip("-")
                        if x_flag_bare:
                            unknown_parser.add_argument(
                                x_flag,
                                nargs="?",
                                default=None,
                                required=False,
                            )
                            unknown_name_map[x_flag_bare] = x_flag

                parsed_unknown, unknown_args = unknown_parser.parse_known_args(
                    args=unknown_args,
                )

                extra_kwargs: Dict[str, str] = {
                    unknown_name_map[k]: v for k, v in vars(parsed_unknown).items()
                }
                if accepts_kwargs:
                    func_kwargs.update(extra_kwargs)
                if accepts_extra_kwargs:
                    func_kwargs["_extra_kwargs"] = extra_kwargs

            if accepts_args:
                func_args = unknown_args

            if accepts_extra_args:
                func_kwargs["_extra_args"] = unknown_args

        else:
            model_inst = parser.parse_args_to_model(
                args=args,
                args_model=args_model,
            )

        return func(*func_args, **vars(model_inst), **func_kwargs)

    @classmethod
    def _build_parser(
        cls,
        command: Optional[Callable[..., Any]] = None,
        subcommands: Optional[Sequence[Callable[..., Any]]] = None,
        named_subcommands: Optional[Dict[str, Callable[..., Any]]] = None,
        **kwargs: Any,
    ) -> "ArgumentParser":
        parser: ArgumentParser = cls(**kwargs)

        if command:
            root_arg_model = make_dataclass_from_func(command)
            cls._set_description_from_docstring(
                parser=parser,
                args_model=root_arg_model,
            )
            parser.add_arguments(root_arg_model)
            parser.set_defaults(
                **{
                    cls.ROOT_FUNC_ATTR_NAME: command,
                    cls.ROOT_FUNC_ARGS_ATTR_NAME: root_arg_model,
                    cls.CMD_FUNC_ATTR_NAME: command,
                    cls.CMD_FUNC_ARGS_ATTR_NAME: root_arg_model,
                },
            )
        else:
            parser.set_defaults(
                **{
                    cls.ROOT_FUNC_ARGS_ATTR_NAME: None,
                    cls.ROOT_FUNC_ATTR_NAME: None,
                    cls.CMD_FUNC_ARGS_ATTR_NAME: None,
                    cls.CMD_FUNC_ATTR_NAME: None,
                },
            )

        if subcommands:
            if callable(subcommands):
                subcommands = [subcommands]
            for x in subcommands:
                parser._register_command(x)

        if named_subcommands:
            for name, x in named_subcommands.items():
                parser._register_command(x, name=name)

        return parser

    @classmethod
    def _run(
        cls,
        *parser_args: Any,
        args: Optional[List[str]] = None,
        default_args: Optional[List[str]] = None,
        **parser_kwargs: Any,
    ) -> Any:
        """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

        Args:
            *parser_args:
            args:
            default_args:
            **parser_kwargs:


        Examples:
            >>> import yapx
            ...
            >>> def print_nums(*args):
            ...     print('Args: ', *args)
            ...
            >>> def find_evens(*args):
            ...     return [x for x in args if int(x) % 2 == 0]
            ...
            >>> def find_odds(*args):
            ...     return [x for x in args if int(x) % 2 != 0]
            ...
            >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
            >>> yapx.run(print_nums, [find_evens, find_odds], args=cli_args)
            Args:  1 2 3 4 5
            ['1', '3', '5']
        """

        parser: ArgumentParser = cls._build_parser(*parser_args, **parser_kwargs)

        if args is None:
            args = sys.argv[1:]

        if not args and default_args:
            args = default_args

        known_args: argparse.Namespace
        known_args, _ = parser.parse_known_args(args)
        parsed_args: Dict[str, Any] = vars(known_args)

        root_func: Optional[Callable[..., Any]] = parsed_args.get(
            cls.ROOT_FUNC_ATTR_NAME,
        )
        root_func_args_model: Optional[Type[Any]] = parsed_args.get(
            cls.ROOT_FUNC_ARGS_ATTR_NAME,
        )
        command_name: Optional[str] = parsed_args.get(cls.CMD_ATTR_NAME)
        command_func: Optional[Callable[..., Any]] = parsed_args.get(
            cls.CMD_FUNC_ATTR_NAME,
        )
        command_func_args_model: Optional[Type[Any]] = parsed_args.get(
            cls.CMD_FUNC_ARGS_ATTR_NAME,
        )

        relay_value: Any = None
        root_result: Any = None

        if root_func:
            root_result = cls._run_func(
                parser=parser,
                func=root_func,
                args_model=root_func_args_model,
                linked_func=command_func,
                args=args,
            )

        if try_isinstance(root_result, GeneratorType):
            try:
                relay_value = next(root_result)
            except StopIteration:
                relay_value = root_result
        else:
            relay_value = root_result

        try:
            if command_name and command_func:
                relay_value = cls._run_func(
                    parser=parser,
                    func=command_func,
                    args_model=command_func_args_model,
                    args=args,
                    linked_func=root_func,
                    relay_value=relay_value,
                )
        finally:
            if try_isinstance(root_result, GeneratorType):
                for gen_result in root_result:
                    if not command_func:
                        relay_value = gen_result

        return relay_value
