import os
import re
import sys
from enum import Enum, auto
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple
from unittest import mock

import pytest
from _pytest.capture import CaptureFixture, CaptureResult

import yapx
import yapx.argument_parser
from yapx.exceptions import MutuallyExclusiveArgumentError

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated


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


def test_run_noargs(capsys: CaptureFixture):
    # 1. ARRANGE
    expected: str = "hello world"
    cli_args: List[str] = []

    # 2. ACT
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    assert captured.out.strip() == expected


def test_run_default(capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["--text", text]
    expected: List[str] = ["hello " + text]
    not_expected: List[str] = ["howdy"]

    # 2. ACT
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup, example_subcmd)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_command(capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["example-subcmd", "--name", text, "--upper"]
    expected: List[str] = [f"howdy {text}".upper()]
    not_expected: List[str] = ["hello"]

    # 2. ACT
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run_command(example_subcmd)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_subcmd(capsys: CaptureFixture):
    # 1. ARRANGE
    text: str = "donald"
    cli_args: List[str] = ["example-subcmd", "--name", text, "--upper"]
    expected: List[str] = [f"howdy {text}".upper()]
    not_expected: List[str] = ["hello"]

    # 2. ACT
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(None, example_subcmd)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_both(capsys: CaptureFixture):
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
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup, example_subcmd)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_generator(capsys: CaptureFixture):
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
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup_generator, example_subcmd)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_args(capsys: CaptureFixture):
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
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup, example_subcmd, _args=cli_args)

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_kwargs_alias(capsys: CaptureFixture):
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
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(
            example_setup,
            **{"command-alias": example_subcmd},
        )

    # 3. ASSERT
    captured: CaptureResult = capsys.readouterr()
    assert captured.out
    for e in expected:
        assert e in captured.out
    for ne in not_expected:
        assert ne not in captured.out


def test_run_kwargs_alias2(capsys: CaptureFixture):
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
    with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
        yapx.run(example_setup, command_alias=example_subcmd)

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
    with mock.patch.object(
        yapx.argument_parser,
        "is_pydantic_available",
        mock.Mock(return_value=use_pydantic),
    ), mock.patch.object(
        yapx.argument_parser.sys,
        "argv",
        [""] + cli_args,
    ):
        result: List[Any] = yapx.run(_func)

        # 3. ASSERT
        assert result == expected


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_enum(use_pydantic: bool):
    # 1. ARRANGE
    class MyEnum(Enum):
        one = auto()
        two = auto()
        three = auto()

    def _func(
        value: Optional[MyEnum],
        value_seq: Optional[Sequence[MyEnum]],
        value_default: Optional[MyEnum] = MyEnum.two,
        value_seq_default: Optional[Sequence[MyEnum]] = yapx.arg(
            default=lambda: [MyEnum.three],
        ),
        value_int_default: Optional[Sequence[int]] = yapx.arg(
            default=lambda: [3, 2, 1],
        ),
    ) -> Tuple[Any, ...]:
        return value, value_seq, value_default, value_seq_default, value_int_default

    cli_args = ["--value", "one", "--value-seq", "three", "two", "one"]

    expected: Tuple[Any, ...] = (
        MyEnum.one,
        [MyEnum.three, MyEnum.two, MyEnum.one],
        MyEnum.two,
        [MyEnum.three],
        [3, 2, 1],
    )

    # 2. ACT
    with mock.patch.object(
        yapx.argument_parser,
        "is_pydantic_available",
        mock.Mock(return_value=use_pydantic),
    ), mock.patch.object(
        yapx.argument_parser.sys,
        "argv",
        [""] + cli_args,
    ):
        result: List[Any] = yapx.run(_func)

    # 3. ASSERT
    assert result

    for i in range(len(expected)):
        assert result[i] == expected[i]


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_path(use_pydantic: bool):
    def _func(
        value: Optional[Path],
        value_seq: Optional[Sequence[Path]],
        value_default: Optional[Path] = Path.cwd(),
        value_seq_default: Optional[Sequence[Path]] = yapx.arg(
            default=lambda: [Path.cwd()],
        ),
    ) -> Tuple[Any, ...]:
        return value, value_seq, value_default, value_seq_default

    cli_args = [
        "--value",
        ".",
        "--value-seq",
        str(Path.cwd()),
        str(Path.cwd().parent),
        str(Path.home()),
    ]

    expected: Tuple[Any, ...] = (
        Path("."),
        [Path.cwd(), Path.cwd().parent, Path.home()],
        Path.cwd(),
        [Path.cwd()],
    )

    # 2. ACT
    with mock.patch.object(
        yapx.argument_parser,
        "is_pydantic_available",
        mock.Mock(return_value=use_pydantic),
    ), mock.patch.object(
        yapx.argument_parser.sys,
        "argv",
        [""] + cli_args,
    ):
        result: List[Any] = yapx.run(_func)

    # 3. ASSERT
    assert result

    for i in range(len(expected)):
        assert result[i] == expected[i]


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
    with mock.patch.object(
        yapx.argument_parser,
        "is_pydantic_available",
        mock.Mock(return_value=use_pydantic),
    ), mock.patch.object(
        yapx.argument_parser.sys,
        "argv",
        [""] + cli_args,
    ):
        if not use_pydantic:
            with pytest.raises(yapx.exceptions.UnsupportedTypeError):
                yapx.run(_func)
        else:
            result: List[Any] = yapx.run(_func)

            # 3. ASSERT
            assert result == expected


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
    with mock.patch.object(
        yapx.argument_parser,
        "is_pydantic_available",
        mock.Mock(return_value=use_pydantic),
    ), mock.patch.object(
        yapx.argument_parser.sys,
        "argv",
        [""] + cli_args,
    ):
        if not use_pydantic:
            with pytest.raises(yapx.exceptions.UnsupportedTypeError):
                yapx.run(_func)
        else:
            result: List[Any] = yapx.run(_func)

            # 3. ASSERT
            assert result == expected


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

    def _setup(*args, _extra_args: Optional[List[str]] = None, **kwargs) -> str:
        assert args == tuple(_extra_args)
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


def test_annotated():
    # 1. ARRANGE
    expected_value: int = 123
    extra_args: List[str] = ["what", "in", "the", "world"]

    def _setup(
        *args,
        value: Annotated[int, yapx.arg(expected_value)],
        value2: int = yapx.arg(expected_value),
        value3: Annotated[Optional[int], yapx.arg(expected_value)] = None,
        value4: Annotated[float, yapx.arg()] = expected_value,
    ) -> str:
        return args, value, value2, value3, value4

    # 2. ACT
    result = yapx.run(
        _setup,
        _args=extra_args,
    )

    # 3. ASSERT
    assert result[0] == tuple(extra_args)

    for x in result[1:]:
        assert x == expected_value


@pytest.mark.parametrize("use_pydantic", [False, True])
def test_run_everything(use_pydantic: bool):
    def _setup():
        return "hello_relay"

    def _func(
        *args: str,
        _extra_args: List[str],
        v1,
        v2: str,
        v3: Annotated[str, yapx.arg("hello_v3")],
        v4: str = "hello_v4",
        v5: str = yapx.arg("hello_v5"),
        #
        v6: Optional[int] = None,
        v7: Optional[Annotated[int, yapx.arg(7)]] = None,
        v8: int = 8,
        v9: int = yapx.arg(9),
        #
        v10: Optional[float] = None,
        v11: Optional[Annotated[float, yapx.arg(3.11)]] = None,
        v12: float = 3.12,
        v13: float = yapx.arg(3.13),
        #
        v14: Optional[bool] = None,
        v15: Optional[Annotated[bool, yapx.arg(False)]] = None,
        v16: Optional[Annotated[bool, yapx.arg(True, action="store_false")]] = None,
        v17: bool = False,
        v18: bool = yapx.arg(False),
        #
        v19: Optional[List[float]] = None,
        v20: Optional[Annotated[List[float], yapx.arg(lambda: [3.2])]] = None,
        v21: List[float] = lambda: [3.21],
        v22: List[float] = yapx.arg(lambda: [3.22]),
        #
        v23: Optional[Sequence[float]] = None,
        v24: Optional[Annotated[Sequence[float], yapx.arg(lambda: [3.2])]] = None,
        v25: Sequence[float] = lambda: [3.21],
        v26: Sequence[float] = yapx.arg(lambda: [3.22]),
        #
        v27: Optional[Tuple[float, ...]] = None,
        v28: Optional[Annotated[Tuple[float, ...], yapx.arg(lambda: (3.2,))]] = None,
        v29: Tuple[float, ...] = lambda: (3.21,),
        v30: Tuple[float, ...] = yapx.arg(lambda: (3.22,)),
        #
        v31: Optional[Set[float]] = None,
        v32: Optional[Annotated[Set[float], yapx.arg(lambda: {3.2})]] = None,
        v33: Set[float] = lambda: {3.21},
        v34: Set[float] = yapx.arg(lambda: {3.22}),
        #
        v35: Optional[Dict[str, float]] = None,
        v36: Optional[
            Annotated[Dict[str, float], yapx.arg(lambda: {"hello": 3.2})]
        ] = None,
        v37: Dict[str, float] = lambda: {"hello": 3.21},
        v38: Dict[str, float] = yapx.arg(lambda: {"hello": 3.22}),
        #
        _relay_value: Any = None,
        **kwargs: Optional[str],
    ) -> None:
        assert args
        assert args == tuple(_extra_args)
        assert args[0] in kwargs
        assert _relay_value == "hello_relay"

        assert v1 == "hello_v1"
        assert v2 == "hello_v2"
        assert v3 == "hello_v3"
        assert v4 == "hello_v4"
        assert v5 == "hello_v5"

        assert v6 == 6
        assert v7 == 7
        assert v8 == 8
        assert v9 == 9

        assert v10 == 3.1
        assert v11 == 3.11
        assert v12 == 3.12
        assert v13 == 3.13

        assert v14 is True
        assert v15 is False
        assert v16 is False
        assert v17 is False
        assert v18 is False

        # list
        assert v19 == [3.19, 3.19, 3.192]
        assert v20 == [3.2]
        assert v21 == [3.21]
        assert v22 == [3.22]

        # sequence
        assert v23 == [3.19, 3.19, 3.192]
        assert v24 == [3.2]
        assert v25 == [3.21]
        assert v26 == [3.22]

        # tuple
        assert v27 == (3.19, 3.19, 3.192)
        assert v28 == (3.2,)
        assert v29 == (3.21,)
        assert v30 == (3.22,)

        # set
        assert v31 == {3.19, 3.192}
        assert v32 == {3.2}
        assert v33 == {3.21}
        assert v34 == {3.22}

        # dict
        assert v35 == {"hello": 3.19, "world": 3.192}
        assert v36 == {"hello": 3.2}
        assert v37 == {"hello": 3.21}
        assert v38 == {"hello": 3.22}

    cli_args: List[str] = [
        "func",
        "--v1",
        "hello_v1",
        "--v2",
        "hello_v2",
        "--v6",
        "6",
        "--v10",
        "3.10",
        "--v14",
        "--v16",
        "--v19",
        "3.19",
        "3.19",
        "3.192",
        "--v23",
        "3.19,",
        "3.19,",
        "3.192",
        "--v27",
        "3.19,",
        "3.19,",
        "3.192",
        "--v31",
        "3.19,",
        "3.19,",
        "3.192",
        "--v35",
        "hello=3.19,",
        "world=3.192",
        "--",
        "purposefully_extra",
        "purposefully_hello=world",
    ]

    # 2. ACT
    try:
        if not use_pydantic:
            mock.patch.object(
                yapx.argument_parser.create_pydantic_model_from_dataclass,
                attribute="__module__",
                new_callable=mock.PropertyMock(return_value="yapx.argument_parser"),
            ).start()

        with mock.patch.object(yapx.argument_parser.sys, "argv", [""] + cli_args):
            yapx.run(_setup, _func)
    finally:
        mock.patch.stopall()
