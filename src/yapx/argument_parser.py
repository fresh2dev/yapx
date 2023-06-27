import argparse
import collections.abc
import sys
from collections import defaultdict
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
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from .__version__ import __version__
from .actions import (
    YapxAction,
    YapxBooleanAction,
    counting_action,
    featflag_action,
    split_csv,
    split_csv_to_dict,
    split_csv_to_set,
    split_csv_to_tuple,
)
from .arg import (
    ARGPARSE_ARG_METADATA_KEY,
    ArgparseArg,
    _eval_type,
    convert_to_command_string,
    convert_to_flag_string,
    make_dataclass_from_func,
)
from .exceptions import NoArgsModelError, raise_unsupported_type_error
from .types import Dataclass, NoneType
from .utils import (
    CommandSchema,
    Trogon,
    add_argument_to,
    build_trogon_schema,
    cast_type,
    create_pydantic_model_from_dataclass,
    is_dataclass_type,
    is_pydantic_available,
    is_shtab_available,
    is_tui_available,
    try_isinstance,
    try_issubclass,
)

__all__ = ["ArgumentParser", "run", "run_commands"]


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 9):
    from typing import _AnnotatedAlias
else:
    from typing_extensions import _AnnotatedAlias

if sys.version_info >= (3, 10):
    from types import UnionType  # pylint: disable=unused-import # noqa: F401

T = TypeVar("T")


class ArgumentParser(argparse.ArgumentParser):
    COMMAND_ATTRIBUTE_NAME: str = "_command"
    FUNC_ATTRIBUTE_NAME: str = "_func"
    ARGS_ATTRIBUTE_NAME: str = "_args_model"

    def __init__(
        self,
        *args: Any,
        prog: Optional[str] = None,
        description: Optional[str] = None,
        add_help: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            prog=prog,
            description=description,
            add_help=add_help,
            **kwargs,
        )

        if is_shtab_available():
            add_argument_to(self, "--print-shell-completion")

        self.kv_separator = "="

        self._mutually_exclusive_args: Dict[
            Optional[Callable[..., Any]],
            Dict[str, List[Tuple[str, Optional[str]]]],
        ] = defaultdict(lambda: defaultdict(list))

        self._inner_type_conversions: Dict[str, Union[type, Callable[[str], Any]]] = {}

        self._tui_schemas: Optional[Dict[str, CommandSchema]] = (
            {} if is_tui_available() else None
        )

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
            >>> parser.set_defaults(_func=lambda x, y: x+y)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            ...
            >>> type(parsed)
            <class 'argparse.Namespace'>
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._args_model(x=parsed.x, y=parsed.y)
            AddNums(x=1, y=2)
            >>> parsed._func(x=parsed.x, y=parsed.y)
            3

            >>> import yapx
            ...
            >>> def add_nums(x: int, y: int):
            ...     return x + y
            ...
            >>> parser = yapx.ArgumentParser()
            >>> parser.add_arguments(add_nums)
            >>> parser.set_defaults(_func=add_nums)
            >>> parsed = parser.parse_args(['-x', '1', '-y', '2'])
            ...
            >>> type(parsed)
            <class 'argparse.Namespace'>
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._args_model(x=parsed.x, y=parsed.y)
            Dataclass_add_nums(x=1, y=2)
            >>> parsed._func(x=parsed.x, y=parsed.y)
            3
        """
        added_args: List[ArgparseArg] = self._add_arguments(self, args_model)

        if self._tui_schemas is not None:
            tui_schema: CommandSchema = build_trogon_schema(
                name=self.prog,
                description=self.description,
                args=added_args,
            )
            self._tui_schemas[tui_schema.name] = tui_schema

    def add_command(
        self,
        name: str,
        args_model: Optional[Union[Callable[..., Any], Type[Dataclass]]] = None,
        no_docstring_description: bool = False,
        add_help: bool = True,
        **kwargs: Any,
    ) -> argparse.ArgumentParser:
        """Create a new subcommand and add arguments from the given function or dataframe to it.

        Args:
            name: name of the command
            args_model: a function or dataclass from which to derive arguments.
            no_docstring_description: if True, do not use use the docstrings as the description
            add_help: add `--help` flag to this command

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
            >>> subparser_1 = parser.add_command('add', AddNums)
            >>> subparser_1.set_defaults(_func=lambda x, y: x+y)
            >>> subparser_2 = parser.add_command('subtract', AddNums)
            >>> subparser_2.set_defaults(_func=lambda x, y: x-y)
            ...
            >>> parsed = parser.parse_args(['add', '-x', '1', '-y', '2'])
            ...
            >>> type(parsed)
            <class 'argparse.Namespace'>
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._args_model(x=parsed.x, y=parsed.y)
            AddNums(x=1, y=2)
            >>> parsed._func(x=parsed.x, y=parsed.y)
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
            >>> subparser_1 = parser.add_command('add', add_nums)
            >>> subparser_1.set_defaults(_func=add_nums)
            >>> subparser_2 = parser.add_command('subtract', subtract_nums)
            >>> subparser_2.set_defaults(_func=subtract_nums)
            ...
            >>> parsed = parser.parse_args(['subtract', '-x', '1', '-y', '2'])
            ...
            >>> type(parsed)
            <class 'argparse.Namespace'>
            >>> (parsed.x, parsed.y)
            (1, 2)
            >>> parsed._args_model(x=parsed.x, y=parsed.y)
            Dataclass_subtract_nums(x=1, y=2)
            >>> parsed._func(x=parsed.x, y=parsed.y)
            -1
        """
        subparsers: argparse.Action = self._get_or_add_subparsers()

        # pylint: disable=protected-access
        assert isinstance(subparsers, argparse._SubParsersAction)
        prog: Optional[str] = kwargs.pop("prog", None)
        if prog:
            prog += " " + name

        description_txt: Optional[str] = None
        if not no_docstring_description and args_model:
            description_txt = self._extract_description_from_docstring(args_model)

        # subparsers must populate 'help' in order to
        # be included in shell-completion.
        # ref: https://github.com/iterative/shtab/issues/54#issuecomment-940516963
        parser = subparsers.add_parser(
            name,
            prog=prog,
            add_help=add_help,
            help=description_txt.splitlines()[0] if description_txt else "",
            **kwargs,
        )
        assert isinstance(parser, argparse.ArgumentParser)

        if description_txt:
            self._set_parser_description(parser=parser, description=description_txt)

        added_args: Optional[List[ArgparseArg]] = (
            self._add_arguments(parser, args_model) if args_model else None
        )

        if self._tui_schemas is not None:
            tui_schema: CommandSchema = build_trogon_schema(
                name=name,
                description=description_txt,
                args=added_args,
            )
            self._tui_schemas[tui_schema.name] = tui_schema

        return parser

    def _register_funcs(
        self,
        *args: Optional[Callable[..., Any]],
        subparser_kwargs: Optional[Dict[str, Any]] = None,
        no_docstring_description: bool = False,
        **kwargs: Callable[..., Any],
    ) -> None:
        def _register_func(
            func: Callable[..., Any],
            name: Optional[str] = None,
            **sp_kwargs: Any,
        ) -> None:
            name = convert_to_command_string(name if name else func.__name__)
            parser: argparse.ArgumentParser = self.add_command(
                name=name,
                args_model=func,
                no_docstring_description=no_docstring_description,
                **sp_kwargs,
            )
            parser.set_defaults(**{self.FUNC_ATTRIBUTE_NAME: func})

        if not subparser_kwargs:
            subparser_kwargs = {}

        for f in args:
            if f:
                _register_func(f, **subparser_kwargs)

        for nm, f in kwargs.items():
            _register_func(f, name=nm, **subparser_kwargs)

    @staticmethod
    def _is_attribute_inherited(dcls: Type[Dataclass], attr: str) -> bool:
        return attr not in dcls.__annotations__

    @classmethod
    def _add_arguments(
        cls,
        parser: argparse.ArgumentParser,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
    ) -> List[ArgparseArg]:
        """
        Derived from: https://github.com/mivade/argparse_dataclass
        """

        model: Type[Dataclass]

        if is_dataclass_type(args_model):
            model = args_model
        else:
            model = make_dataclass_from_func(args_model)

        if parser.get_default(cls.ARGS_ATTRIBUTE_NAME):
            err: str = "This parser already contains arguments from another dataclass."
            parser.error(err)

        parser.set_defaults(**{cls.ARGS_ATTRIBUTE_NAME: model})

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
                            argparse_argument.action = split_csv_to_dict

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
                                argparse_argument.action = split_csv_to_set
                            elif fld_type_origin is tuple:
                                argparse_argument.action = split_csv_to_tuple
                            else:
                                argparse_argument.action = split_csv

                    elif not is_pydantic_available():
                        raise_unsupported_type_error(fld_type)

                    if try_issubclass(fld_type, Enum):
                        argparse_argument.choices = [x.name for x in fld_type]

                    if isinstance(parser, cls):
                        # store desired types for casting later
                        # pylint: disable=protected-access
                        parser._inner_type_conversions[
                            argparse_argument.dest
                        ] = partial(cast_type, target_type=fld_type)

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
                else [x for x in option_strings if x.startswith(parser.prefix_chars)]
            )

            required: bool = kwargs["required"]

            if not args:
                # positional arg
                del kwargs["required"]
                if not required and kwargs.get("nargs") is None:
                    kwargs["nargs"] = "?"
                kwargs["type"] = partial(cast_type, target_type=fld_type)
            elif argparse_argument.type is bool:
                for k in "type", "nargs", "const", "choices", "metavar":
                    kwargs.pop(k, None)
                if not kwargs["action"]:
                    kwargs["action"] = YapxBooleanAction
            elif argparse_argument.nargs != 0:
                kwargs["type"] = partial(cast_type, target_type=fld_type)
            elif fld_type is int:
                # 'int' args with nargs==0 are "counting" parameters (-vvv).
                kwargs["action"] = counting_action
            elif fld_type is str:
                # 'str' args with nargs==0 are "feature flag" parameters.
                kwargs["action"] = featflag_action

            # if given `default` cast it to the expected type.
            if (
                fld.default is not MISSING
                or fld.default_factory is not MISSING
                or kwargs.get("default") is not None
            ):
                if kwargs.get("action") and try_issubclass(
                    kwargs["action"],
                    (YapxAction, YapxBooleanAction),
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
            if required:
                help_msg_parts.append("Required")
            else:
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
                kwargs["help"] += " " + help_msg
            else:
                kwargs["help"] = help_msg

            group: Optional[str] = kwargs.pop("group", None)

            if not group:
                group = "Arguments"

            arg_group: argparse._ArgumentGroup = parser_arg_groups.get(group, None)

            if not arg_group:
                arg_group = parser.add_argument_group(group)
                parser_arg_groups[group] = arg_group

            if argparse_argument.exclusive:
                if required:
                    err: str = (
                        "A mutually-exclusive argument cannot be required:"
                        f" {argparse_argument.dest}"
                    )
                    parser.error(err)

                parser._mutually_exclusive_args[
                    parser._defaults.get(cls.FUNC_ATTRIBUTE_NAME)
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
        elif not is_pydantic_available():
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
            and not is_pydantic_available()
        ):
            raise_unsupported_type_error(type_container_subtype_origin)

        return type_container_subtype

    def _get_subparsers(self) -> Optional[argparse._SubParsersAction]:
        for a in self._actions:
            if isinstance(
                a,
                argparse._SubParsersAction,  # pylint: disable=protected-access
            ):
                return a
        return None

    def _get_or_add_subparsers(self) -> argparse._SubParsersAction:
        subparsers = self._get_subparsers()

        if not subparsers:
            subparsers = self.add_subparsers(dest=self.COMMAND_ATTRIBUTE_NAME)

        return subparsers

    def _post_parse_args(  # type: ignore[override]
        self,
        namespace: argparse.Namespace,
    ) -> argparse.Namespace:
        # delete vars created by shtab
        sh_complete_attr = "print_shell_completion"
        if hasattr(namespace, sh_complete_attr):
            delattr(namespace, sh_complete_attr)

        func_mx_arg_groups: Dict[
            str,
            List[Tuple[str, Optional[str]]],
        ] = self._mutually_exclusive_args.get(
            getattr(namespace, self.FUNC_ATTRIBUTE_NAME, None),
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

        return namespace

    def parse_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        parsed: argparse.Namespace = super().parse_args(args=args, namespace=namespace)
        return self._post_parse_args(parsed)

    def parse_known_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[argparse.Namespace, List[str]]:
        parsed: argparse.Namespace
        unknown: List[str]
        parsed, unknown = super().parse_known_args(args=args, namespace=namespace)
        return self._post_parse_args(parsed), unknown

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
            >>> type(parsed)
            <class 'yapx.argument_parser.AddNums'>
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
            >>> type(parsed)
            <class 'yapx.argument_parser.AddNums'>
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
            args_model = parsed_args.get(self.ARGS_ATTRIBUTE_NAME)
            if not args_model:
                raise NoArgsModelError

        args_union: Dict[str, Any] = self._union_args_with_model(
            args_dict=parsed_args,
            args_model=args_model,
        )

        if not skip_pydantic_validation and is_pydantic_available():
            # pylint: disable=not-callable
            args_union = vars(
                create_pydantic_model_from_dataclass(args_model)(**args_union),
            )

        return args_model(**args_union)

    @staticmethod
    def _union_args_with_model(
        args_dict: Dict[str, Any],
        args_model: Type[Dataclass],
    ) -> Dict[str, Any]:
        return {k: v for k, v in args_dict.items() if k in args_model.__annotations__}

    def print_help(
        self,
        file: Optional[IO[str]] = None,
        full: bool = False,
    ) -> None:
        """Print CLI help.

        Args:
            full: if True, print help for all commands.

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
            >>> parser.print_help(full=True)  #doctest: +SKIP
        """
        super().print_help(file)

        if full:
            subparsers: Optional[argparse._SubParsersAction] = self._get_subparsers()
            separator: str = "\n" + ("*" * 80)
            if subparsers:
                for choice, subparser in subparsers.choices.items():
                    print(separator)
                    print(f">>> {choice}")
                    subparser.print_help(file)

    @classmethod
    def _set_parser_description(
        cls,
        parser: argparse.ArgumentParser,
        description: Optional[str],
    ):
        if not description:
            return

        parser.description = description

        if "\n" in description:
            # allow newlines in parser description
            parser.formatter_class = argparse.RawTextHelpFormatter

    @staticmethod
    def _extract_description_from_docstring(
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

    def _show_tui(self) -> None:
        assert self._tui_schemas

        root_schema: CommandSchema = self._tui_schemas.pop(
            self.prog,
            build_trogon_schema(
                name=self.prog,
                description=self.description,
            ),
        )

        Trogon.from_schemas(
            root_schema,
            *self._tui_schemas.values(),
            app_name=self.prog,
            app_version=__version__,
        ).run()

    @classmethod
    def _run_func(
        cls,
        parser: "ArgumentParser",
        func: Callable[..., Any],
        args_model: Type[Any],
        args: List[str],
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
    def _run(
        cls,
        *args: Optional[Callable[..., Any]],
        _args: Optional[List[str]] = None,
        _prog: Optional[str] = None,
        _print_help: bool = False,
        _help_flags: Optional[List[Optional[str]]] = None,
        _tui_flags: Optional[List[Optional[str]]] = None,
        _no_docstring_description: bool = False,
        **kwargs: Callable[..., Any],
    ) -> Any:
        """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

        *args: 1st arg is root-command; subsequent args are sub-commands.
        **kwargs: sub-commands with specific names.
        _args: arguments to parse (default=sys.argv[1:]).
        _prog: name of the program.
        _print_help: will print help and exit.
        _help_flags: list of flags that activate help (default=['-h', '--help']).
            Provide an empty list `[]` to disable help.
        _tui_flags: list of flags that activate the TUI (default=['--tui']).
            Provide a list containing 'None' `[None]` to activate the TUI when no args are given.
        _no_docstring_description: do not use docstrings for command descriptions

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
            >>> yapx.run(print_nums, find_evens, find_odds, _args=cli_args)
            Args:  1 2 3 4 5
            ['1', '3', '5']
        """

        parser_shared_kwargs: Dict[str, Any] = {
            "prog": _prog,
            "add_help": _help_flags is None,
        }

        parser: ArgumentParser = cls(**parser_shared_kwargs)

        if _help_flags:
            if isinstance(_help_flags, str):
                _help_flags = [_help_flags]
            _help_flags = [
                convert_to_flag_string(x) if x is not None else x for x in _help_flags
            ]
            parser.add_argument(
                *[x for x in _help_flags if x],
                default=argparse.SUPPRESS,
                action="help",
                help="Show this help message and exit.",
            )

        if _tui_flags is None:
            _tui_flags = ["--tui"]
        elif _tui_flags:
            if isinstance(_tui_flags, str):
                _tui_flags = [_tui_flags]
            _tui_flags = [
                convert_to_flag_string(x) if x is not None else x for x in _tui_flags
            ]

        parser.set_defaults(
            **{cls.ARGS_ATTRIBUTE_NAME: None, cls.FUNC_ATTRIBUTE_NAME: None},
        )

        setup_func: Optional[Callable[..., Any]] = None
        setup_func_arg_model: Optional[Type[Any]] = None
        cmd_funcs: List[Callable[..., Any]] = []

        if args:
            setup_func = args[0]

            if setup_func:
                setup_func_arg_model = make_dataclass_from_func(setup_func)
                if not _no_docstring_description:
                    cls._set_parser_description(
                        parser=parser,
                        description=cls._extract_description_from_docstring(
                            setup_func_arg_model,
                        ),
                    )
                parser.add_arguments(setup_func_arg_model)

            _cmd_funcs: Iterable[Optional[Callable[..., Any]]] = args[1:]
            assert isinstance(_cmd_funcs, tuple)
            cmd_funcs.extend(_cmd_funcs)

        parser._register_funcs(
            *cmd_funcs,
            subparser_kwargs=parser_shared_kwargs,
            no_docstring_description=_no_docstring_description,
            **kwargs,
        )

        if _args is None:
            _args = sys.argv[1:]

        _show_tui: bool = False

        if not _print_help:
            if not _args:
                # auto-show tui/help if no args given and 'None' is in _flags
                _show_tui = None in _tui_flags and is_tui_available()
                _print_help = not _show_tui and _help_flags and None in _help_flags
            else:
                if _help_flags:
                    _print_help = (
                        f"{[convert_to_flag_string(x) for x in _help_flags if x][-1]}-full"
                        in _args
                    )
                else:
                    _print_help = "--help-full" in _args

                if not _print_help:
                    _show_tui = any(x in _args for x in _tui_flags)

        if _print_help:
            if _tui_flags:
                parser.add_argument(
                    *[x for x in _tui_flags if x],
                    action="store_true",
                    help="Show the terminal user interface (TUI).",
                )
            parser.print_help(full=True)
            return None
        elif _show_tui:
            if not is_tui_available():
                parser.error("Missing 'trogon' library. Try `pip install yapx[tui]`")
            parser._show_tui()
            return None

        known_args: argparse.Namespace
        known_args, _ = parser.parse_known_args(_args)
        parsed_args: Dict[str, Any] = vars(known_args)

        # parsed_args.get(cls.COMMAND_ATTRIBUTE_NAME)

        func: Optional[Callable[..., Any]] = parsed_args.get(cls.FUNC_ATTRIBUTE_NAME)
        args_model: Optional[Type[Any]] = parsed_args.get(cls.ARGS_ATTRIBUTE_NAME)
        relay_value: Any = None

        setup_result: Any = None
        if setup_func:
            assert setup_func_arg_model
            setup_result = cls._run_func(
                parser=parser,
                func=setup_func,
                args_model=setup_func_arg_model,
                linked_func=func,
                args=_args,
            )

        if try_isinstance(setup_result, GeneratorType):
            try:
                relay_value = next(setup_result)
            except StopIteration:
                relay_value = setup_result
        else:
            relay_value = setup_result

        try:
            if func:
                relay_value = cls._run_func(
                    parser=parser,
                    func=func,
                    args_model=args_model,
                    args=_args,
                    linked_func=setup_func,
                    relay_value=relay_value,
                )
        finally:
            if try_isinstance(setup_result, GeneratorType):
                for gen_result in setup_result:
                    if not func:
                        relay_value = gen_result

        return relay_value


def run(
    *args: Optional[Callable[..., Any]],
    _args: Optional[List[str]] = None,
    _prog: Optional[str] = None,
    _print_help: bool = False,
    _help_flags: Optional[List[Optional[str]]] = None,
    _tui_flags: Optional[List[Optional[str]]] = None,
    _no_docstring_description: bool = False,
    **kwargs: Callable[..., Any],
) -> Any:
    """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

    Args:
        *args: 1st arg is root-command; subsequent args are sub-commands.
        **kwargs: sub-commands with specific names.
        _args: arguments to parse (default=sys.argv[1:]).
        _prog: name of the program.
        _print_help: will print help and exit.
        _help_flags: list of flags that activate help (default=['-h', '--help']).
            Provide an empty list `[]` to disable help.
        _tui_flags: list of flags that activate the TUI (default=['--tui']).
            Provide a list containing 'None' `[None]` to activate the TUI when no args are given.
        _no_docstring_description: do not use docstrings for command descriptions

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
        >>> yapx.run(print_nums, find_evens, find_odds, _args=cli_args)
        Args:  1 2 3 4 5
        ['1', '3', '5']
    """
    # pylint: disable=protected-access
    return ArgumentParser._run(
        *args,
        _args=_args,
        _prog=_prog,
        _print_help=_print_help,
        _help_flags=_help_flags,
        _tui_flags=_tui_flags,
        _no_docstring_description=_no_docstring_description,
        **kwargs,
    )


def run_commands(
    *args: Optional[Callable[..., Any]],
    _args: Optional[List[str]] = None,
    _prog: Optional[str] = None,
    _print_help: bool = False,
    _help_flags: Optional[List[Optional[str]]] = None,
    _tui_flags: Optional[List[Optional[str]]] = None,
    _no_docstring_description: bool = False,
    **kwargs: Callable[..., Any],
) -> Any:
    """Use given functions to construct a CLI, parse the args, and invoke the appropriate command.

    Args:
        *args: functions to add as sub-commands.
        **kwargs: sub-commands with specific names.
        _args: arguments to parse (default=sys.argv[1:]).
        _prog: name of the program.
        _print_help: will print help and exit.
        _help_flags: list of flags that activate help (default=['-h', '--help']).
            Provide an empty list `[]` to disable help.
        _tui_flags: list of flags that activate the TUI (default=['--tui']).
            Provide a list containing 'None' `[None]` to activate the TUI when no args are given.
        _no_docstring_description: do not use docstrings for command descriptions

    Examples:
        >>> import yapx
        ...
        >>> def find_evens(*args):
        ...     return [x for x in args if int(x) % 2 == 0]
        ...
        >>> def find_odds(*args):
        ...     return [x for x in args if int(x) % 2 != 0]
        ...
        >>> cli_args = ['find-odds', '1', '2', '3', '4', '5']
        >>> yapx.run_commands(find_evens, find_odds, _args=cli_args)
        ['1', '3', '5']
    """
    # pylint: disable=protected-access
    return ArgumentParser._run(
        None,
        *args,
        _args=_args,
        _prog=_prog,
        _print_help=_print_help,
        _help_flags=_help_flags,
        _tui_flags=_tui_flags,
        _no_docstring_description=_no_docstring_description,
        **kwargs,
    )
