from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest
from _pytest.capture import CaptureFixture, CaptureResult

import yapx
from yapx.types import Annotated, Dataclass


def _common_test(args: List[str], args_model: Dataclass, expected):
    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(args_model)

    parsed_args: Dict[str, Any] = vars(parser.parse_args(args))

    # 3. ASSERT
    assert parsed_args["values"] == expected


@pytest.mark.parametrize(
    "annotation",
    [
        Optional[List[str]],
        Optional[Sequence[str]],
        Optional[Tuple[str, ...]],
        Optional[Set[str]],
        Optional[Dict[str, str]],
    ],
)
def test_split_no_args(annotation):
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[annotation, yapx.arg(None, nargs="*")]

    # 2. ACT
    _common_test(args=[], args_model=ArgsModel, expected=None)


def test_split_csv_empty():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[List[str], yapx.arg(nargs="*")]

    cli_args = ["--values"]
    expected: List[Optional[str]] = []

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_set_empty():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Set[str], yapx.arg(nargs="*")]

    cli_args = ["--values"]
    expected: Set = set()

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_tuple_empty():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Tuple[str], yapx.arg(nargs="*")]

    cli_args = ["--values"]
    expected: Tuple = tuple()

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_dict_empty():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Dict[str, str], yapx.arg(nargs="*")]

    cli_args = ["--values"]
    expected: Dict = {}

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_csv():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[List[str], yapx.arg(nargs="+")]

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

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_csv_to_set():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Set[str], yapx.arg(nargs="+")]

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

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_csv_to_tuple():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Tuple[str, ...], yapx.arg(nargs="+")]

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

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_split_csv_to_dict():
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: Annotated[Dict[str, str], yapx.arg(nargs="+")]

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
        "1,",
        "1,",
        "1,",
        "1,",
        "2,",
        "3,",
        " 4 = hello,",
        "5,",
        "6  = world,",
        " 7 = ,",
        " 8 9 ,",
    ]

    # 2. ACT
    _common_test(args=cli_args, args_model=ArgsModel, expected=expected)


def test_print_help_full(capsys: CaptureFixture):
    # 1. ARRANGE
    @dataclass
    class ArgsModel:
        values: str = yapx.arg(env="TEST_VALUES")

    cli_args: List[str] = ["-h", "--values", "1"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)
    parser.add_command(ArgsModel, name="subcmd1")
    parser.add_command(ArgsModel, name="subcmd2")
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
    parser.add_command(ArgsModel, name=expected_cmd)
    parser.add_command(ArgsModel, name="subcmd2")
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
        "--values",
        "--help",
        "hello world",
        "TEST_VALUES",
    ]

    cli_args: List[str] = ["-h"]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(ArgsModel)
    parser.add_command(ArgsModel, name=expected_cmd)
    parser.add_command(ArgsModel, name="subcmd2")
    with pytest.raises(SystemExit) as pexit:
        parser.parse_args(cli_args)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert not captured.err
    assert expected_cmd in captured.out
    assert pexit.value.code == 0

    for e in expected_txt:
        assert e in captured.out
