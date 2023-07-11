import argparse
from dataclasses import dataclass
from typing import Any, Dict

import yapx


def test_add_command() -> None:
    # 1. ARRANGE
    @dataclass
    class RootModel:
        debug: bool

    @dataclass
    class CmdModel(RootModel):
        str_value: str
        int_value: int
        bool_value: bool

    expected_cmd_name: str = "my-command"

    expected: Dict[str, Any] = {
        "debug": True,
        yapx.ArgumentParser.CMD_ATTR_NAME: expected_cmd_name,
        yapx.ArgumentParser.CMD_FUNC_ARGS_ATTR_NAME: CmdModel,
        "str_value": "abc",
        "int_value": 5,
        "bool_value": True,
    }

    cli_args = [
        "--debug",
        expected_cmd_name,
        "--str-value",
        "abc",
        "--int-value",
        "5",
        "--bool-value",
    ]

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(RootModel)
    cmd_parser: argparse.ArgumentParser = parser.add_command(
        CmdModel,
        name=expected_cmd_name,
    )

    args: Dict[str, Any] = vars(parser.parse_args(cli_args))

    # 3. ASSERT
    # pylint: disable=protected-access
    assert cmd_parser
    assert expected_cmd_name in parser._get_or_add_subparsers().choices
    assert args == expected
