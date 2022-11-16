import argparse
import collections.abc
import sys
from collections import defaultdict
from dataclasses import MISSING, Field, fields
from typing import (
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

from .actions import (
    print_help,
    split_csv,
    split_csv_to_dict,
    split_csv_to_set,
    split_csv_to_tuple,
)
from .arg import (
    ARGPARSE_ARG_METADATA_KEY,
    ArgparseArg,
    convert_to_command_string,
    convert_to_flag_string,
    convert_to_short_flag_string,
    make_dataclass_from_func,
)
from .argparse_action import YapxAction
from .types import Dataclass, NoneType
from .utils import is_dataclass_type, str2bool

__all__ = ["ArgumentParser", "run", "run_command"]


try:
    from pydantic.dataclasses import create_pydantic_model_from_dataclass
except ModuleNotFoundError:

    def create_pydantic_model_from_dataclass():
        ...


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

T = TypeVar("T")


class ArgumentParser(argparse.ArgumentParser):

    COMMAND_ATTRIBUTE_NAME: str = "_command"
    FUNC_ATTRIBUTE_NAME: str = "_func"
    ARGS_ATTRIBUTE_NAME: str = "_args_model"

    def __init__(self, *args: Any, **kwargs: Any):
        add_help: bool = bool(kwargs.pop("add_help", True))
        super().__init__(*args, add_help=False, **kwargs)

        if add_help:
            parser_help_args = self.add_argument_group("help")
            parser_help_args.add_argument(
                "-h",
                "--help",
                action=print_help,
                default=argparse.SUPPRESS,
                help=argparse.SUPPRESS,
            )

        self._inner_type_conversions: Dict[str, Union[type, Callable[[str], Any]]] = {}

    def add_arguments(
        self, args_model: Union[Callable[..., Any], Type[Dataclass]]
    ) -> None:
        self._add_arguments(self, args_model)

    def add_command(
        self,
        name: str,
        args_model: Optional[Union[Callable[..., Any], Type[Dataclass]]] = None,
        use_docstr_description: Optional[bool] = True,
        **kwargs: Any,
    ) -> argparse.ArgumentParser:
        subparsers: argparse.Action = self._get_or_add_subparsers()
        # pylint: disable=protected-access
        assert isinstance(subparsers, argparse._SubParsersAction)
        prog: Optional[str] = kwargs.pop("prog", None)
        if prog:
            prog += " " + name
        add_help: bool = bool(kwargs.pop("add_help", True))
        parser = subparsers.add_parser(name, prog=prog, add_help=add_help, **kwargs)
        assert isinstance(parser, argparse.ArgumentParser)
        if args_model:
            self._add_arguments(parser, args_model)
            if use_docstr_description:
                self._docstring_to_description(parser=parser, args_model=args_model)
        return parser

    def _register_funcs(
        self,
        *args: Optional[Callable[..., Any]],
        subparser_kwargs: Optional[Dict[str, Any]] = None,
        use_docstr_description: Optional[bool] = True,
        **kwargs: Callable[..., Any],
    ) -> None:
        def _register_func(
            func: Callable[..., Any],
            name: Optional[str] = None,
            **sp_kwargs: Any,
        ) -> None:
            name = convert_to_command_string(name if name else func.__name__)
            parser: argparse.ArgumentParser = self.add_command(
                name,
                func,
                use_docstr_description,
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
    ) -> None:
        """
        Derived from: https://github.com/mivade/argparse_dataclass

        License
        -------
        MIT License

        Copyright (c) 2021 Michael V. DePalatis and contributors

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """

        model: Type[Dataclass]

        if is_dataclass_type(args_model):
            model = args_model
        else:
            model = make_dataclass_from_func(args_model)

        if parser.get_default(cls.ARGS_ATTRIBUTE_NAME):
            raise Exception("This parser already has args from another dataclass")
        parser.set_defaults(**{cls.ARGS_ATTRIBUTE_NAME: model})

        parser_required_args: Dict[str, argparse._ArgumentGroup] = defaultdict(
            lambda: parser.add_argument_group("required arguments")
        )
        parser_optional_args: Dict[str, argparse._ArgumentGroup] = defaultdict(
            lambda: parser.add_argument_group("optional arguments")
        )
        parser_exclusive_args: Dict[str, argparse._ArgumentGroup] = defaultdict(
            parser.add_mutually_exclusive_group
        )
        # parser_required_args = parser.add_argument_group("required arguments")
        # parser_optional_args = parser.add_argument_group("optional arguments")
        # parser_exclusive_args = parser.add_mutually_exclusive_group()
        parser_arg_groups: Dict[str, argparse._ArgumentGroup] = {}

        novel_fields: List[Field] = list(
            filter(
                lambda f: not cls._is_attribute_inherited(dcls=model, attr=f.name),
                fields(model),
            )
        )

        novel_field_args: List[Tuple[Field, ArgparseArg]] = [
            (x, x.metadata.get(ARGPARSE_ARG_METADATA_KEY, ArgparseArg()))
            for x in novel_fields
        ]

        registered_flags: List[str] = ["-h", "--help"]

        for fld, argparse_argument in novel_field_args:
            if argparse_argument.option_strings:
                if isinstance(argparse_argument.option_strings, str):
                    argparse_argument.option_strings = [
                        argparse_argument.option_strings
                    ]
                for x in argparse_argument.option_strings:
                    registered_flags.append(x)

        for fld, argparse_argument in novel_field_args:
            argparse_argument.dest = fld.name

            if not argparse_argument.option_strings and not argparse_argument.pos:
                long_flag: str = convert_to_flag_string(fld.name)
                short_flag: str = convert_to_short_flag_string(long_flag)
                arg_flags: List[str] = []

                for flg in short_flag, long_flag:
                    if flg not in registered_flags:
                        arg_flags.append(flg)

                if not arg_flags:
                    raise ValueError(f"Derived flag name already in use: {long_flag}")

                argparse_argument.option_strings = arg_flags

            if argparse_argument.option_strings:
                registered_flags.extend(argparse_argument.option_strings)

            if fld.default is not MISSING:
                argparse_argument.default = fld.default
                argparse_argument.required = False
            elif fld.default_factory is not MISSING:
                argparse_argument.default = fld.default_factory()
                argparse_argument.required = False
            else:
                argparse_argument.default = None
                argparse_argument.required = True

            if not argparse_argument.required and argparse_argument.dest.startswith(
                "_"
            ):
                # skip private arg
                continue

            fld_type: Union[str, Type[Any]] = fld.type

            # basic support for handling deferred annotation evaluation,
            # when using `from __future__ import annotations`
            if isinstance(fld_type, str):
                fld_type = cls._eval_fld_type(fld_type)
                assert not isinstance(fld_type, str)

            if cls._get_type_origin(fld_type) is Union:
                fld_type = cls._extract_type_from_container(fld_type)

            help_type: str = (
                fld_type.__name__
                if isinstance(fld_type, type)
                else str(fld_type).split(".")[-1]
            )

            fld_type_origin: Optional[Type[Any]] = cls._get_type_origin(fld_type)
            if fld_type_origin:
                if fld_type_origin is Literal:
                    argparse_argument.choices = list(cls._get_type_args(fld_type))
                    if argparse_argument.choices:
                        fld_type = type(argparse_argument.choices[0])
                else:
                    # this is a type-container (List, Set, Tuple, Dict, ...)
                    if issubclass(fld_type_origin, collections.abc.Mapping):
                        fld_type = cls._extract_type_from_container(
                            fld_type, assert_primitive=True
                        )
                        if not argparse_argument.action:
                            argparse_argument.action = split_csv_to_dict
                    elif (
                        issubclass(fld_type_origin, collections.abc.Iterable)
                        or fld_type_origin is set
                    ):
                        fld_type = cls._extract_type_from_container(
                            fld_type, assert_primitive=True
                        )
                        if not argparse_argument.action:
                            if fld_type_origin is set:
                                argparse_argument.action = split_csv_to_set
                            elif fld_type_origin is tuple:
                                argparse_argument.action = split_csv_to_tuple
                            else:
                                argparse_argument.action = split_csv
                    elif not cls.is_pydantic_available():
                        raise TypeError(f"Unsupported type: {fld_type_origin.__name__}")

                    # store desired types for casting later
                    if isinstance(parser, cls):
                        if fld_type in (str, int, float):
                            # pylint: disable=protected-access
                            parser._inner_type_conversions[
                                argparse_argument.dest
                            ] = fld_type
                        elif fld_type is bool:
                            # pylint: disable=protected-access
                            parser._inner_type_conversions[
                                argparse_argument.dest
                            ] = str2bool
                    # type-containers must only contain strings
                    # until parsed by argparse.
                    fld_type = str

                    argparse_argument.nargs = "+" if argparse_argument.required else "*"

            if fld_type in (str, int, float, bool):
                argparse_argument.type = fld_type
            else:
                argparse_argument.type = str

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
                if not required:
                    kwargs["nargs"] = "?"

            if kwargs["type"] is bool:
                if not args:
                    kwargs["type"] = str2bool
                else:
                    for k in "type", "nargs", "const", "choices", "metavar":
                        kwargs.pop(k, None)
                    required = False
                    kwargs["required"] = False
                    if fld.default is not True:
                        if kwargs["default"] is None:
                            kwargs["default"] = False
                        kwargs["action"] = "store_true"
                    else:
                        kwargs["action"] = "store_false"

            help_msg: str = f"type: {help_type}"
            # pylint: disable=protected-access
            if not required:
                help_default = kwargs.get("default")
                if isinstance(help_default, str):
                    help_default = f"'{help_default}'"
                help_msg += f", default: {help_default}"

            if argparse_argument._env_var:
                help_msg += f", env: {argparse_argument._env_var}"
            help_msg = f"({help_msg})"

            if kwargs.get("help"):
                kwargs["help"] += " " + help_msg
            else:
                kwargs["help"] = help_msg

            exclusive: Optional[bool] = kwargs.pop("exclusive", False)
            group: Optional[str] = kwargs.pop("group", None)

            if exclusive:
                parser_exclusive_args["exclusive"].add_argument(*args, **kwargs)
            elif group:
                arg_group: argparse._ArgumentGroup = parser_arg_groups.get(
                    group, parser.add_argument_group(group)
                )
                arg_group.add_argument(*args, **kwargs)
                parser_arg_groups[group] = arg_group
            elif required:
                parser_required_args["required"].add_argument(*args, **kwargs)
            else:
                parser_optional_args["optional"].add_argument(*args, **kwargs)

    @classmethod
    def _eval_fld_type(cls, fld_type: str) -> Any:
        if "[" in fld_type:
            # None | list[str] --> None | List[str]
            fld_type = "|".join(
                (y.capitalize() if "[" in y else y)
                for x in fld_type.split("|")
                for y in [x.strip()]
            )
        if "|" in fld_type:
            # None | str --> Union[None, str]
            fld_type = f"Union[{fld_type.replace('|', ',')}]"

        return eval(fld_type)  # pylint: disable=eval-used

    @staticmethod
    def _get_type_origin(t: Type[Any]) -> Optional[Type[Any]]:
        return getattr(t, "__origin__", None)

    @staticmethod
    def _get_type_args(t: Type[Any]) -> Tuple[Type[Any], ...]:
        return getattr(t, "__args__", ())

    @classmethod
    def _extract_type_from_container(
        cls, type_container: Type[Any], assert_primitive: Optional[bool] = False
    ) -> Type[Any]:
        type_container_origin: Any = cls._get_type_origin(type_container)

        if type_container_origin is None:
            raise TypeError(
                f"Given type is not a container:" f" {type_container.__name__}"
            )

        type_container_args = cls._get_type_args(type_container)

        results: List[Type[Any]] = [
            a for a in type_container_args if a is not ... and a is not NoneType
        ]

        type_container_subtype: type

        if (
            type_container_origin is Union
            or issubclass(type_container_origin, collections.abc.Sequence)
            or type_container_origin is set
        ):
            if len(results) != 1:
                results.clear()
            else:
                type_container_subtype = results[0]
        elif issubclass(type_container_origin, collections.abc.Mapping):
            if len(results) != 2:
                results.clear()
            else:
                key_type, value_type = results
                if key_type is not str:
                    raise TypeError("Dictionary keys must be type `str`")
                type_container_subtype = value_type
        elif not cls.is_pydantic_available():
            raise TypeError(f"Unsupported type: {type_container.__name__}")

        if not results:
            raise TypeError("Too many types in container")

        type_container_subtype_origin = cls._get_type_origin(type_container_subtype)

        if type_container_subtype_origin is Union:
            type_container_subtype = cls._extract_type_from_container(
                type_container_subtype
            )
            type_container_subtype_origin = cls._get_type_origin(type_container_subtype)

        if (
            assert_primitive
            and type_container_subtype_origin
            and not cls.is_pydantic_available()
        ):
            raise TypeError(f"Unsupported type: {str(type_container_subtype_origin)}")

        return type_container_subtype

    @classmethod
    def _get_subparsers(
        cls, parser: argparse.ArgumentParser
    ) -> Optional[argparse._SubParsersAction]:
        # pylint: disable=protected-access
        for a in parser._actions:
            # pylint: disable=protected-access
            if isinstance(a, argparse._SubParsersAction):
                return a
        return None

    def _get_or_add_subparsers(self) -> argparse._SubParsersAction:
        subparsers = self._get_subparsers(parser=self)

        if not subparsers:
            return self.add_subparsers(dest=self.COMMAND_ATTRIBUTE_NAME)

        return subparsers

    def parse_args(  # type: ignore[override]
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        """parse args, and be sure to run all actions,
        even if arg was not specified (when value is default)
        see: https://stackoverflow.com/a/21588198
        """
        parsed: argparse.Namespace = super().parse_args(args=args, namespace=namespace)

        for a in self._actions:
            if (
                isinstance(a, YapxAction)
                and hasattr(parsed, a.dest)
                and getattr(parsed, a.dest) == a.default
            ):
                a(parser=self, namespace=parsed, values=a.default)

        return parsed

    @staticmethod
    def is_pydantic_available() -> bool:
        return create_pydantic_model_from_dataclass.__module__ != __name__

    def parse_known_args_to_model(
        self,
        args: Optional[Sequence[str]] = None,
        args_model: Optional[Type[Dataclass]] = None,
        use_pydantic: Optional[bool] = True,
    ) -> Tuple[Dataclass, List[str]]:
        parsed_args: argparse.Namespace
        unknown_args: List[str]
        parsed_args, unknown_args = self.parse_known_args(args)

        return (
            self._parse_args_to_model(
                args=parsed_args, args_model=args_model, use_pydantic=use_pydantic
            ),
            unknown_args,
        )

    def parse_args_to_model(
        self,
        args: Optional[Sequence[str]] = None,
        args_model: Optional[Type[Dataclass]] = None,
        use_pydantic: Optional[bool] = True,
    ) -> Dataclass:
        parsed_args: argparse.Namespace = self.parse_args(args)
        return self._parse_args_to_model(
            args=parsed_args, args_model=args_model, use_pydantic=use_pydantic
        )

    def _parse_args_to_model(
        self,
        args: argparse.Namespace,
        args_model: Optional[Type[Dataclass]] = None,
        use_pydantic: Optional[bool] = True,
    ) -> Dataclass:
        parsed_args: Dict[str, Any] = vars(args)

        if not args_model:
            args_model = parsed_args.get(self.ARGS_ATTRIBUTE_NAME)
            if not args_model:
                raise Exception("No arg model provided")

        args_union: Dict[str, Any] = self._union_args_with_model(
            args_dict=parsed_args, args_model=args_model
        )

        if use_pydantic:
            if not self.is_pydantic_available():
                # print("pydantic is not installed")
                pass
            else:
                args_union = vars(
                    create_pydantic_model_from_dataclass(args_model)(**args_union)
                )

        return args_model(**args_union)

    @staticmethod
    def _union_args_with_model(
        args_dict: Dict[str, Any], args_model: Type[Dataclass]
    ) -> Dict[str, Any]:
        return {k: v for k, v in args_dict.items() if k in args_model.__annotations__}

    @classmethod
    def _print_help(
        cls, parser: argparse.ArgumentParser, include_all: Optional[bool] = False
    ) -> None:
        separator: str = "*" * 80
        print(separator)
        print(parser.format_help())

        if include_all:
            subparsers: Optional[argparse._SubParsersAction] = cls._get_subparsers(
                parser=parser
            )
            if subparsers:
                for choice, subparser in subparsers.choices.items():
                    print(separator)
                    print(f">>> {choice}")
                    print(subparser.format_help())

    def print_help_all(self) -> None:
        self._print_help(parser=self, include_all=True)

    @staticmethod
    def _docstring_to_description(
        parser: argparse.ArgumentParser,
        args_model: Union[Callable[..., Any], Type[Dataclass]],
    ) -> None:
        docstr_lines: Optional[List[str]] = None
        if args_model.__doc__:
            docstr_lines = args_model.__doc__.strip().splitlines()

        if not docstr_lines:
            return

        text_block_ends_at: int = 0
        for i, line in enumerate(docstr_lines):
            if not line.strip():
                text_block_ends_at = i
                break

        if text_block_ends_at > 0:
            docstr_lines = docstr_lines[:text_block_ends_at]

        parser.description = "\n".join([x.strip() for x in docstr_lines])

        if len(docstr_lines) > 1:
            # allow newlines in parser description
            parser.formatter_class = argparse.RawTextHelpFormatter

    @classmethod
    def _run(
        cls,
        *args: Optional[Callable[..., Any]],
        _args: Optional[List[str]] = None,
        _prog: Optional[str] = None,
        _use_pydantic: Optional[bool] = True,
        _print_help: Optional[bool] = False,
        _use_docstr_description: Optional[bool] = True,
        **kwargs: Callable[..., Any],
    ) -> Any:
        parser_shared_kwargs: Dict[str, Any] = {"prog": _prog}

        parser: ArgumentParser = cls(**parser_shared_kwargs)

        parser.set_defaults(
            **{cls.ARGS_ATTRIBUTE_NAME: None, cls.FUNC_ATTRIBUTE_NAME: None}
        )

        setup_func: Optional[Callable[..., Any]] = None
        setup_func_arg_model: Optional[Type[Any]] = None
        cmd_funcs: List[Callable[..., Any]] = []

        if args:
            setup_func = args[0]

            if setup_func:
                setup_func_arg_model = make_dataclass_from_func(setup_func)
                parser.add_arguments(setup_func_arg_model)
                if _use_docstr_description:
                    cls._docstring_to_description(
                        parser=parser,
                        args_model=setup_func_arg_model,
                    )

            _cmd_funcs: Iterable[Optional[Callable[..., Any]]] = args[1:]
            assert isinstance(_cmd_funcs, tuple)
            cmd_funcs.extend(_cmd_funcs)

        parser._register_funcs(
            *cmd_funcs,
            subparser_kwargs=parser_shared_kwargs,
            use_docstr_description=_use_docstr_description,
            **kwargs,
        )

        if _print_help:
            parser.print_help_all()
            parser.exit()

        if not _args:
            _args = sys.argv[1:]

        parsed_args: Dict[str, Any] = vars(parser.parse_args(_args))

        # parsed_args.get(cls.COMMAND_ATTRIBUTE_NAME)

        func: Optional[Callable[..., Any]] = parsed_args.get(cls.FUNC_ATTRIBUTE_NAME)

        args_model: Optional[Type[Any]] = parsed_args.get(cls.ARGS_ATTRIBUTE_NAME)

        result: Any = None

        for f, m in [(setup_func, setup_func_arg_model), (func, args_model)]:
            if f:
                model_inst: Dataclass = parser.parse_args_to_model(
                    args=_args, args_model=m, use_pydantic=_use_pydantic
                )
                f_kwargs: Dict[str, Any] = vars(model_inst)
                result = f(**f_kwargs)

        return result


def run(
    *args: Optional[Callable[..., Any]],
    _args: Optional[List[str]] = None,
    _prog: Optional[str] = None,
    _use_pydantic: Optional[bool] = True,
    _print_help: Optional[bool] = False,
    **kwargs: Callable[..., Any],
) -> Any:
    # pylint: disable=protected-access
    return ArgumentParser._run(
        *args,
        _args=_args,
        _prog=_prog,
        _use_pydantic=_use_pydantic,
        _print_help=_print_help,
        **kwargs,
    )


def run_command(
    *args: Callable[..., Any],
    _args: Optional[List[str]] = None,
    _prog: Optional[str] = None,
    _use_pydantic: Optional[bool] = True,
    _print_help: Optional[bool] = False,
    **kwargs: Callable[..., Any],
) -> Any:
    return run(
        None,
        *args,
        _args=_args,
        _prog=_prog,
        _use_pydantic=_use_pydantic,
        _print_help=_print_help,
        **kwargs,
    )
