from __future__ import annotations

import yapx


def test_add_arguments_deferred():
    # 1. ARRANGE

    # pylint: disable=unused-argument
    # noinspection Annotator
    def func(
        value: None | list[str] = yapx.arg(lambda: ["hello world"], nargs="*"),
        value_kv: None | dict[str, str] = yapx.arg(None, nargs="*"),
    ):
        ...

    cli_args: list[str] = ["--value-kv", "hello=world", "this=that"]

    expected_value: list[str] = ["hello world"]
    expected_value_kv: dict[str, str] = {"hello": "world", "this": "that"}

    # 2. ACT
    parser: yapx.ArgumentParser = yapx.ArgumentParser()
    parser.add_arguments(func)
    args: type[yapx.types.Dataclass] = parser.parse_args_to_model(cli_args)

    # 3. ASSERT
    assert args
    assert hasattr(args, "value")
    assert args.value == expected_value
    assert hasattr(args, "value_kv")
    assert args.value_kv == expected_value_kv
