import sys
from argparse import Action
from dataclasses import MISSING, dataclass, make_dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type

import pytest

import yapx
from yapx.arg import ArgparseArg

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


@pytest.mark.parametrize(
    ("name", "type_annotation", "default", "expected"),
    [
        (
            "a_str",
            str,
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-str"],
                type=str,
                help="(type: str)",
                required=True,
            ),
        ),
        (
            "a_str",
            str,
            "hello",
            ArgparseArg(
                option_strings=["-a", "--a-str"],
                type=str,
                help="(type: str, default: 'hello')",
                required=False,
            ),
        ),
        (
            "a_str",
            Optional[str],
            None,
            ArgparseArg(
                option_strings=["-a", "--a-str"],
                type=str,
                help="(type: str, default: None)",
                required=False,
            ),
        ),
        # int
        (
            "a_int",
            int,
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-int"],
                type=int,
                help="(type: int)",
                required=True,
            ),
        ),
        (
            "a_int",
            int,
            123,
            ArgparseArg(
                option_strings=["-a", "--a-int"],
                type=int,
                help="(type: int, default: 123)",
                required=False,
            ),
        ),
        (
            "a_int",
            Optional[int],
            None,
            ArgparseArg(
                option_strings=["-a", "--a-int"],
                type=int,
                help="(type: int, default: None)",
                required=False,
            ),
        ),
        # float
        (
            "a_float",
            float,
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-float"],
                type=float,
                help="(type: float)",
                required=True,
            ),
        ),
        (
            "a_float",
            float,
            3.14,
            ArgparseArg(
                option_strings=["-a", "--a-float"],
                type=float,
                help="(type: float, default: 3.14)",
                required=False,
            ),
        ),
        # bool
        (
            "a_bool",
            bool,
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-bool"],
                type=None,
                help="(type: bool, default: False)",
                nargs=0,
                const=True,
                required=False,
                default=False,
            ),
        ),
        # bool (default=True)
        (
            "a_bool",
            bool,
            True,
            ArgparseArg(
                option_strings=["-a", "--a-bool"],
                type=None,
                help="(type: bool, default: True)",
                nargs=0,
                const=False,
                required=False,
            ),
        ),
        # list
        (
            "a_list",
            List[str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-list"],
                type=str,
                help="(type: List[str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_list",
            List[str],
            list,
            ArgparseArg(
                option_strings=["-a", "--a-list"],
                type=str,
                help="(type: List[str], default: [])",
                nargs="*",
                required=False,
            ),
        ),
        # tuple
        (
            "a_tuple",
            Tuple[str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-tuple"],
                type=str,
                help="(type: Tuple[str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_tuple",
            Tuple[str],
            tuple,
            ArgparseArg(
                option_strings=["-a", "--a-tuple"],
                type=str,
                help="(type: Tuple[str], default: ())",
                nargs="*",
                required=False,
            ),
        ),
        # sequence
        (
            "a_sequence",
            Sequence[str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-sequence"],
                type=str,
                help="(type: Sequence[str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_sequence",
            Sequence[str],
            list,
            ArgparseArg(
                option_strings=["-a", "--a-sequence"],
                type=str,
                help="(type: Sequence[str], default: [])",
                nargs="*",
                required=False,
            ),
        ),
        # set
        (
            "a_set",
            Set[str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-set"],
                type=str,
                help="(type: Set[str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_set",
            Set[str],
            set,
            ArgparseArg(
                option_strings=["-a", "--a-set"],
                type=str,
                help="(type: Set[str], default: set())",
                nargs="*",
                required=False,
            ),
        ),
        # dict
        (
            "a_dict",
            Dict[str, str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-dict"],
                type=str,
                help="(type: Dict[str, str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_dict",
            Dict[str, str],
            dict,
            ArgparseArg(
                option_strings=["-a", "--a-dict"],
                type=str,
                help="(type: Dict[str, str], default: {})",
                nargs="*",
                required=False,
            ),
        ),
        # mapping
        (
            "a_mapping",
            Mapping[str, str],
            MISSING,
            ArgparseArg(
                option_strings=["-a", "--a-mapping"],
                type=str,
                help="(type: Mapping[str, str])",
                nargs="+",
                required=True,
            ),
        ),
        (
            "a_mapping",
            Mapping[str, str],
            dict,
            ArgparseArg(
                option_strings=["-a", "--a-mapping"],
                type=str,
                help="(type: Mapping[str, str], default: {})",
                nargs="*",
                required=False,
            ),
        ),
        (
            "a_literal",
            Literal["one", "two", "three"],
            "two",
            ArgparseArg(
                option_strings=["-a", "--a-literal"],
                type=str,
                help="(type: Literal['one', 'two', 'three'], default: 'two')",
                choices=["one", "two", "three"],
                required=False,
            ),
        ),
        (
            "a_literal_int",
            Literal[1, 2, 3],
            2,
            ArgparseArg(
                option_strings=["-a", "--a-literal-int"],
                type=int,
                help="(type: Literal[1, 2, 3], default: 2)",
                choices=[1, 2, 3],
                required=False,
            ),
        ),
    ],
)
def test_add_arguments(
    name: str, type_annotation: Type[Any], default: Any, expected: ArgparseArg
):
    # 1. ARRANGE
    expected.dest = name

    argfield_kwargs: Dict[str, Any] = {}
    if default is not MISSING:
        if default and callable(default):
            expected.default = default()
        else:
            expected.default = default
        argfield_kwargs["default"] = default

    ArgsModel: Type[yapx.types.Dataclass] = make_dataclass(
        "ArgsModel", [(name, type_annotation, yapx.arg(**argfield_kwargs))]
    )

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_actions: List[Action] = parser._actions[1:]
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
    with pytest.raises(Exception):
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
    parser.add_arguments(ArgsModel)

    result: ArgsModel = parser.parse_args_to_model(cli_args)

    # 3. ASSERT
    assert not result.value
    assert result.conflict == expected


def test_add_arguments_conflict_error():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        help: str

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()

    with pytest.raises(ValueError) as excinfo:
        parser.add_arguments(ArgsModel)

    # 3. ASSERT
    assert excinfo.value


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
    sys.version_info.minor < 10, reason="test modern annotations in Python 3.10+"
)
def test_add_arguments_modern():
    # 1. ARRANGE
    # pylint: disable=unused-argument
    # noinspection Annotator
    def func(value: None | list[str] = yapx.arg(default=lambda: ["hello world"])):
        ...

    expected: list[str] = ["hello world"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(func)
    args: Dict[str, str] = vars(parser.parse_args([]))

    # 3. ASSERT
    assert args
    assert "value" in args
    assert args["value"] == expected
