from argparse import Action
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest
from _pytest.capture import CaptureFixture, CaptureResult

import yapx
from yapx.argparse_action import YapxAction


def test_split_csv():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: List[Optional[str]]

    expected: List[Optional[str]] = [
        "1",
        "1",
        "1",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]
    cli_args = ["--values"] + expected

    expected_action_name: str = "split_csv"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_action: Action = parser._actions[2]
    assert isinstance(parser_action, YapxAction)
    # pylint: disable=protected-access
    assert parser_action.name == expected_action_name

    assert "values" in args
    assert args["values"] == expected


def test_split_csv_to_set():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Set[Optional[str]]

    expected: Set[Optional[str]] = {
        "",
        "1",
        "2",
        "3  4",
        "5",
        "6  7",
        "8",
        "9",
    }
    cli_args = ["--values"] + list(expected)

    expected_action_name: str = "split_csv_to_set"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_action: Action = parser._actions[2]
    assert isinstance(parser_action, YapxAction)
    # pylint: disable=protected-access
    assert parser_action.name == expected_action_name

    assert "values" in args
    assert args["values"] == expected


def test_split_csv_to_tuple():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Tuple[Optional[str], ...]

    expected: Tuple[Optional[str]] = (
        "",
        "",
        "1",
        "1",
        "1",
        "1",
        "2",
        "3  4",
        "5",
        "6  7",
        "8",
        "9",
        "",
    )
    cli_args = ["--values"] + list(expected)

    expected_action_name: str = "split_csv_to_tuple"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_action: Action = parser._actions[2]
    assert isinstance(parser_action, YapxAction)
    # pylint: disable=protected-access
    assert parser_action.name == expected_action_name

    assert "values" in args
    assert args["values"] == expected


def test_split_csv_to_dict():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Dict[str, Optional[str]]

    expected: Dict[str, Optional[str]] = {
        "1": None,
        "2": None,
        "3": None,
        "4": "hello",
        "5": None,
        "6": "world",
        "7": None,
        "8 9": None,
    }
    cli_args = [
        "--values",
        "1",
        "1",
        "1",
        "1",
        "2",
        "3",
        " 4 = hello",
        "5",
        "6  = world ",
        " 7 = ",
        " 8 9 ",
    ]

    expected_action_name: str = "split_csv_to_dict"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_action: Action = parser._actions[2]
    assert isinstance(parser_action, YapxAction)
    # pylint: disable=protected-access
    assert parser_action.name == expected_action_name

    assert "values" in args
    assert args["values"] == expected


def test_str2enum():
    # 1. ARRANGE
    class MyEnum(Enum):
        one = auto()
        two = auto()
        three = auto()

    @dataclass
    class ArgsModel:
        value: Optional[MyEnum]
        value_seq: Optional[Sequence[MyEnum]]
        value_default: Optional[MyEnum] = MyEnum.two
        value_seq_default: Optional[Sequence[MyEnum]] = yapx.arg(
            default=lambda: [MyEnum.three]
        )
        value_int_default: Optional[Sequence[int]] = yapx.arg(default=lambda: [3, 2, 1])

    cli_args = ["--value", "one", "--value-seq", "three", "two", "one"]

    expected: Tuple[MyEnum, List[MyEnum]] = (
        MyEnum.one,
        [MyEnum.three, MyEnum.two, MyEnum.one],
        MyEnum.two,
        [MyEnum.three],
        [3, 2, 1],
    )

    expected_action_name: str = "str2enum"

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    parser_action: Action = parser._actions[2]
    assert isinstance(parser_action, YapxAction)
    # pylint: disable=protected-access
    assert parser_action.name == expected_action_name

    assert "value" in args
    assert args["value"] == expected[0]
    assert "value_seq" in args
    assert args["value_seq"] == expected[1]
    assert "value_default" in args
    assert args["value_default"] == expected[2]
    assert "value_seq_default" in args
    assert args["value_seq_default"] == expected[3]
    assert "value_int_default" in args
    assert args["value_int_default"] == expected[4]


def test_print_help_full(capsys: CaptureFixture):
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: str = yapx.arg(env="TEST_VALUES")

    cli_args: List[str] = ["-h", "--values", "1"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)
    parser.add_command("subcmd1", ArgsModel)
    parser.add_command("subcmd2", ArgsModel)
    with pytest.raises(SystemExit) as pexit:
        parser.parse_args(cli_args)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert not captured.err
    assert captured.out
    assert pexit.value.code == 0


def test_print_help_subparser(capsys: CaptureFixture):
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: str = yapx.arg(env="TEST_VALUES")

    expected_cmd: str = "subcmd1"

    cli_args: List[str] = [expected_cmd, "-h"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)
    parser.add_command(expected_cmd, ArgsModel)
    parser.add_command("subcmd2", ArgsModel)
    with pytest.raises(SystemExit) as pexit:
        parser.parse_args(cli_args)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert not captured.err
    assert expected_cmd in captured.out
    assert pexit.value.code == 0


def test_print_docstring_help(capsys: CaptureFixture):
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        """hello world"""

        values: str = yapx.arg(env="TEST_VALUES")

    expected_cmd: str = "subcmd1"
    expected_txt: List[str] = [
        "{subcmd1,subcmd2}",
        "--values",
        "--help",
        "hello world",
        "TEST_VALUES",
    ]

    cli_args: List[str] = ["-h"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)
    parser.add_command(expected_cmd, ArgsModel)
    parser.add_command("subcmd2", ArgsModel)
    with pytest.raises(SystemExit) as pexit:
        parser.parse_args(cli_args)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert not captured.err
    assert expected_cmd in captured.out
    assert pexit.value.code == 0

    for e in expected_txt:
        assert e in captured.out
