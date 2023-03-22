import os
import re
from ipaddress import IPv4Address
from typing import Any, List, Optional, Pattern
from unittest import mock

import pytest
from _pytest.capture import CaptureFixture, CaptureResult

import yapx
import yapx.argument_parser
from yapx.exceptions import MutuallyExclusiveArgumentError


def example_setup(text: str = "world") -> str:
    print("hello " + text)


def example_setup_generator(text: str = "world") -> str:
    print("hello " + text)
    yield
    print("hallo " + text)
    yield
    yield
    yield


def example_empty_subcmd() -> str:
    ...


def example_subcmd(name: str, upper: Optional[bool]) -> str:
    msg: str = "howdy " + name
    if upper:
        msg = msg.upper()
    print(msg)


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_noargs(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    expected: str = "hello world"
    cli_args: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    assert captured.out.strip() == expected


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_default(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["--text", text]
    expected: List[str] = ["hello " + text]
    not_expected: List[str] = ["howdy"]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup, example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_command(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["example-subcmd", "--name", text, "--upper"]
    expected: List[str] = [f"howdy {text}".upper()]
    not_expected: List[str] = ["hello"]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run_command(example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_subcmd(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["example-subcmd", "--name", text, "--upper"]
    expected: List[str] = [f"howdy {text}".upper()]
    not_expected: List[str] = ["hello"]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(None, example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_both(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = [
        "--text",
        text,
        "example-subcmd",
        "--name",
        text,
        "--upper",
    ]
    expected: List[str] = [f"hello {text}", f"howdy {text}".upper()]
    not_expected: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup, example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_generator(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = [
        "--text",
        text,
        "example-subcmd",
        "--name",
        text,
        "--upper",
    ]
    expected: List[str] = [f"hello {text}", f"howdy {text}".upper(), f"hallo {text}"]
    not_expected: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup_generator, example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_args(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = [
        "--text",
        text,
        "example-subcmd",
        "--name",
        text,
        "--upper",
    ]
    expected: List[str] = [f"hello {text}", f"howdy {text}".upper()]
    not_expected: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup, example_subcmd, _args=cli_args)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_kwargs_alias(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = [
        "--text",
        text,
        "command-alias",
        "--name",
        text,
        "--upper",
    ]
    expected: List[str] = [f"hello {text}", f"howdy {text}".upper()]
    not_expected: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(
                example_setup,
                **{"command-alias": example_subcmd},
            )
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_kwargs_alias2(use_pydantic: bool, capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = [
        "--text",
        text,
        "command-alias",
        "--name",
        text,
        "--upper",
    ]
    expected: List[str] = [f"hello {text}", f"howdy {text}".upper()]
    not_expected: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(example_setup, command_alias=example_subcmd)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_ipv4address(use_pydantic: bool):
    env_var_name: str = "FUNKY_ARG"

    env_values: List[str] = ["127.0.0.1", "192.168.0.1", "9.9.9.9"]
    os.environ[env_var_name] = " LIST[ " + "   ".join(env_values) + " ] "

    expected: List[IPv4Address] = [IPv4Address(ip) for ip in env_values]

    def _func(
        ip_addrs: List[IPv4Address] = yapx.arg(env=env_var_name),
    ) -> List[int]:
        return ip_addrs

    del os.environ[env_var_name]

    # 1. ARRANGE
    cli_args: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

            with mock.patch.object(
                yapx.argument_parser.sys,
                "argv",
                [""] + cli_args,
            ), pytest.raises(yapx.exceptions.UnsupportedTypeError):
                yapx.run(_func)
            return

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            result: List[Any] = yapx.run(_func)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    assert result
    assert isinstance(result, list)

    # if use_pydantic:
    assert result == expected


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_patterns(use_pydantic: bool):
    env_var_name: str = "FUNKY_ARG"

    env_values: List[str] = ["abc", "def", ".*"]
    os.environ[env_var_name] = (
        " LiSt[ " + "   ".join(str(i) for i in env_values) + " ] "
    )

    expected: List[Pattern] = [re.compile(x) for x in env_values]

    def _func(ip_addrs: List[Pattern] = yapx.arg(env=env_var_name)) -> List[int]:
        return ip_addrs

    del os.environ[env_var_name]

    # 1. ARRANGE
    cli_args: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

            with mock.patch.object(
                yapx.argument_parser.sys,
                "argv",
                [""] + cli_args,
            ), pytest.raises(yapx.exceptions.UnsupportedTypeError):
                yapx.run(_func)
            return

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            result: List[Any] = yapx.run(_func)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    assert result
    assert isinstance(result, list)

    if use_pydantic:
        assert result == expected
    else:
        assert result == env_values


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_pattern(use_pydantic: bool):
    env_var_name: str = "FUNKY_ARG"

    env_value: str = r".+"
    os.environ[env_var_name] = env_value

    expected: Pattern = re.compile(env_value)

    def _func(ip_addr: Pattern = yapx.arg(env=env_var_name)) -> Pattern:
        return ip_addr

    del os.environ[env_var_name]

    # 1. ARRANGE
    cli_args: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

            with mock.patch.object(
                yapx.argument_parser.sys,
                "argv",
                [""] + cli_args,
            ), pytest.raises(yapx.exceptions.UnsupportedTypeError):
                yapx.run(_func)
            return

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            result: List[Any] = yapx.run(_func)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    assert result

    if use_pydantic:
        assert result == expected
    else:
        assert result == env_value


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_bools(use_pydantic: bool):
    env_var_name: str = "FUNKY_ARG"

    env_values: List[str] = ["0", "1", "true", "t", "false", "f", "yes", "y", "no", "n"]
    os.environ[env_var_name] = (
        " list[ " + "   ".join(str(i) for i in env_values) + " ] "
    )

    expected: List[bool] = [
        x.lower() in ("1", "true", "t", "yes", "y") for x in env_values
    ]

    def _func(ip_addrs: List[bool] = yapx.arg(env=env_var_name)) -> List[int]:
        return ip_addrs

    del os.environ[env_var_name]

    # 1. ARRANGE
    cli_args: List[str] = []

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            result: List[Any] = yapx.run(_func)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    assert result
    assert isinstance(result, list)

    assert result == expected


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_pos_list(use_pydantic: bool):
    def _func(
        this: Optional[List[str]] = yapx.arg(None, pos=True, exclusive=True),
        that: Optional[List[str]] = yapx.arg(None, exclusive=True),
    ) -> List[int]:
        return this, that

    # 1. ARRANGE
    cli_args: List[str] = ["--that", "world"]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            result: List[Any] = yapx.run(_func)
    finally:
        mock.patch.stopall()

    # 3. ASSERT
    assert result


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_exclusive(use_pydantic: bool):
    def _func(
        this: Optional[List[str]] = yapx.arg(None, pos=True, exclusive=True),
        that: Optional[List[str]] = yapx.arg(None, exclusive=True),
        the: Optional[bool] = yapx.arg(None, exclusive=True),
    ) -> List[int]:
        return this, that, the

    # 1. ARRANGE
    cli_args: List[str] = ["hello", "--that", "world"]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(
            yapx.argument_parser.sys,
            "argv",
            [""] + cli_args,
        ), pytest.raises(MutuallyExclusiveArgumentError):
            yapx.run(_func)
    finally:
        mock.patch.stopall()


def test_print_shell_completion(capsys: CaptureFixture):
    # 1. ARRANGE
    cli_args: List[str] = ["--print-shell-completion", "zsh"]
    expected: List[str] = ["--text", "example-empty-subcmd", "example-subcmd"]
    not_expected: List[str] = []

    # 2. ACT
    with pytest.raises(SystemExit):
        yapx.run(
            example_setup,
            example_empty_subcmd,
            example_subcmd,
            _args=cli_args,
        )

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()

    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_extra_args():
    # 1. ARRANGE
    cli_args: List[str] = ["subcmd", "what", "in", "the", "world=this"]
    expected: List[str] = cli_args[1:]

    def _setup(_extra_args: Optional[List[str]] = None, **kwargs) -> str:
        assert _extra_args[0] in kwargs
        assert kwargs["world"] == "this"
        return _extra_args

    def _subcmd(_relay_value: Any) -> str:
        assert _relay_value == expected
        return _relay_value

    # 2. ACT
    result = yapx.run(
        _setup,
        _subcmd,
        _args=cli_args,
    )

    # 3. ASSERT
    assert result == expected
