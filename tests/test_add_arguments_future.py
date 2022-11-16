from __future__ import annotations

import yapx


def test_add_arguments_deferred():
    # 1. ARRANGE

    # pylint: disable=unused-argument
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
