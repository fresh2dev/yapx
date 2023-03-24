import sys
from argparse import Action, ArgumentError
from dataclasses import MISSING, dataclass, make_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type

import pytest

import yapx
from yapx.arg import ArgparseArg
from yapx.exceptions import ParserClosedError

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 9):
    from typing import Annotated, _AnnotatedAlias
else:
    from typing_extensions import Annotated, _AnnotatedAlias


@pytest.mark.parametrize(
    ("name", "type_annotation", "default", "expected"),
    [
        # str, annotated
        (
            "a_str",
            Annotated[str, "nothingburger"],
            MISSING,
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Required",
                required=True,
            ),
        ),
        (
            "a_str",
            Annotated[str, yapx.arg("hello")],
            MISSING,
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Default: 'hello'",
                required=False,
                default="hello",
            ),
        ),
        (
            "a_str",
            Annotated[Optional[str], yapx.arg("hello")],
            None,
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Default: 'hello'",
                required=False,
                default="hello",
            ),
        ),
        # str
        (
            "a_str",
            str,
            MISSING,
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Required",
                required=True,
            ),
        ),
        (
            "a_str",
            str,
            "hello",
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Default: 'hello'",
                required=False,
                default="hello",
            ),
        ),
        (
            "a_str",
            Optional[str],
            None,
            ArgparseArg(
                option_strings=["--a-str"],
                type=str,
                help="> Type: str, Default: None",
                required=False,
                default=None,
            ),
        ),
        # int
        (
            "a_int",
            int,
            MISSING,
            ArgparseArg(
                option_strings=["--a-int"],
                type=int,
                help="> Type: int, Required",
                required=True,
            ),
        ),
        (
            "a_int",
            int,
            123,
            ArgparseArg(
                option_strings=["--a-int"],
                type=int,
                help="> Type: int, Default: 123",
                required=False,
                default=123,
            ),
        ),
        (
            "a_int",
            Optional[int],
            None,
            ArgparseArg(
                option_strings=["--a-int"],
                type=int,
                help="> Type: int, Default: None",
                required=False,
                default=None,
            ),
        ),
        # float
        (
            "a_float",
            float,
            MISSING,
            ArgparseArg(
                option_strings=["--a-float"],
                type=float,
                help="> Type: float, Required",
                required=True,
            ),
        ),
        (
            "a_float",
            float,
            3.14,
            ArgparseArg(
                option_strings=["--a-float"],
                type=float,
                help="> Type: float, Default: 3.14",
                required=False,
                default=3.14,
            ),
        ),
        # bool
        (
            "a_bool",
            bool,
            MISSING,
            ArgparseArg(
                option_strings=["--a-bool"],
                type=None,
                help="> Type: bool, Default: None",
                nargs=0,
                const=True,
                required=False,
                default=None,
            ),
        ),
        # bool (True)
        (
            "a_bool",
            bool,
            True,
            ArgparseArg(
                option_strings=["--a-bool"],
                type=None,
                help="> Type: bool, Default: True",
                nargs=0,
                const=True,
                required=False,
                default=True,
            ),
        ),
        # list
        (
            "a_list",
            List[str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-list"],
                type=str,
                help="> Type: List[str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_list",
            List[str],
            list,
            ArgparseArg(
                option_strings=["--a-list"],
                type=str,
                help="> Type: List[str], Default: []",
                nargs="*",
                required=False,
                default=[],
            ),
        ),
        # tuple
        (
            "a_tuple",
            Tuple[str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-tuple"],
                type=str,
                help="> Type: Tuple[str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_tuple",
            Tuple[str],
            tuple,
            ArgparseArg(
                option_strings=["--a-tuple"],
                type=str,
                help="> Type: Tuple[str], Default: ()",
                nargs="*",
                required=False,
                default=(),
            ),
        ),
        # sequence
        (
            "a_sequence",
            Sequence[str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-sequence"],
                type=str,
                help="> Type: Sequence[str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_sequence",
            Sequence[str],
            list,
            ArgparseArg(
                option_strings=["--a-sequence"],
                type=str,
                help="> Type: Sequence[str], Default: []",
                nargs="*",
                required=False,
                default=[],
            ),
        ),
        # set
        (
            "a_set",
            Set[str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-set"],
                type=str,
                help="> Type: Set[str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_set",
            Set[str],
            set,
            ArgparseArg(
                option_strings=["--a-set"],
                type=str,
                help="> Type: Set[str], Default: set()",
                nargs="*",
                required=False,
                default=set(),
            ),
        ),
        # dict
        (
            "a_dict",
            Dict[str, str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-dict"],
                type=str,
                help="> Type: Dict[str, str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_dict",
            Dict[str, str],
            dict,
            ArgparseArg(
                option_strings=["--a-dict"],
                type=str,
                help="> Type: Dict[str, str], Default: {}",
                nargs="*",
                required=False,
                default={},
            ),
        ),
        # mapping
        (
            "a_mapping",
            Mapping[str, str],
            MISSING,
            ArgparseArg(
                option_strings=["--a-mapping"],
                type=str,
                help="> Type: Mapping[str, str], Required",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_mapping",
            Mapping[str, str],
            dict,
            ArgparseArg(
                option_strings=["--a-mapping"],
                type=str,
                help="> Type: Mapping[str, str], Default: {}",
                nargs="*",
                required=False,
                default={},
            ),
        ),
        (
            "a_literal",
            Literal["one", "two", "three"],
            "two",
            ArgparseArg(
                option_strings=["--a-literal"],
                type=str,
                help="> Type: Literal['one', 'two', 'three'], Default: 'two'",
                choices=["one", "two", "three"],
                required=False,
                default="two",
            ),
        ),
        (
            "a_literal_int",
            Literal[1, 2, 3],
            2,
            ArgparseArg(
                option_strings=["--a-literal-int"],
                type=int,
                help="> Type: Literal[1, 2, 3], Default: 2",
                choices=[1, 2, 3],
                required=False,
                default=2,
            ),
        ),
    ],
)
def test_add_arguments(
    name: str,
    type_annotation: Type[Any],
    default: Any,
    expected: ArgparseArg,
):
    # 1. ARRANGE
    expected.dest = name

    if not isinstance(type_annotation, _AnnotatedAlias):
        default = yapx.arg(default=default)

    ArgsModel: Type[yapx.types.Dataclass] = make_dataclass(
        "ArgsModel",
        [(name, type_annotation, default)],
    )

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_actions: List[Action] = parser._actions[2:]
    # pylint: disable=no-member
    assert len(parser_actions) == len(ArgsModel.__dataclass_fields__)

    for action in parser_actions:
        assert action.dest

        action_dict: Dict[str, Any] = vars(action)

        expected_dict: Dict[str, Any] = expected.asdict()
        for custom_key in "action", "pos", "group", "exclusive":
            del expected_dict[custom_key]

        for k, v in expected_dict.items():
            assert action_dict[k] == v, f"value of {k} is not equal for '{action.dest}'"


def test_add_arguments_twice():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        value: str

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    # 3. ASSERT
    with pytest.raises(ParserClosedError):
        parser.add_arguments(ArgsModel)


def test_add_arguments_conflict():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        conflict: str = yapx.arg(flags=["--value"])
        value: Optional[str] = None

    expected: str = "hello world"
    cli_args: List[str] = ["--value", expected]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    with pytest.raises(ArgumentError):
        parser.add_arguments(ArgsModel)

    result: ArgsModel = parser.parse_args_to_model(cli_args)

    # 3. ASSERT
    assert not result.value
    assert result.conflict == expected


def test_add_arguments_func():
    # 1. ARRANGE
    # pylint: disable=unused-argument
    def func(value: str = yapx.arg(default="hello world")):
        ...

    expected: str = "hello world"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(func)
    args: Dict[str, Any] = vars(parser.parse_args([]))

    # 3. ASSERT
    assert args
    assert "value" in args
    assert args["value"] == expected


@pytest.mark.skipif(
    sys.version_info.minor < 10,
    reason="test modern annotations in Python 3.10+",
)
def test_add_arguments_modern():
    # pylint: disable=unused-argument,unsupported-binary-operation,unsubscriptable-object
    # 1. ARRANGE
    # noinspection Annotator
    def func(value: None | list[str] = yapx.arg(default=lambda: ["hello world"])):
        ...

    expected: list[str] = ["hello world"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(func)
    args: dict[str, str] = vars(parser.parse_args([]))

    # 3. ASSERT
    assert args
    assert "value" in args
    assert args["value"] == expected
