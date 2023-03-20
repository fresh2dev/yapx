from dataclasses import dataclass
from typing import Optional

import pytest

import yapx


@pytest.mark.parametrize("x", ["1", "true", "t", "yes", "y"])
def test_str2bool_true(x):
    assert yapx.utils.str2bool(x) is True


@pytest.mark.parametrize("x", ["0", "false", "f", "no", "n"])
def test_str2bool_false(x):
    assert yapx.utils.str2bool(x) is False


def test_is_dataclass_type():
    @dataclass
    class Something:
        nothing: Optional[str] = None

    assert yapx.utils.is_dataclass_type(Something)
    assert yapx.utils.is_dataclass_type(Something())


def test_try_isinstance():
    assert yapx.utils.try_isinstance("", str)
    assert not yapx.utils.try_isinstance(0, str)
    assert not yapx.utils.try_isinstance(None, str)


def test_try_issubclass():
    class Something(str):
        ...

    assert yapx.utils.try_issubclass(Something, str)
    assert not yapx.utils.try_issubclass(Something(), str)
    assert not yapx.utils.try_issubclass(None, str)


def test_coalesce():
    assert yapx.utils.coalesce("", "this") == ""
    assert yapx.utils.coalesce("", "this", null_or_empty=True) == "this"
    assert yapx.utils.coalesce(None, "this") == "this"


def test_parse_sequence():
    args, kwargs = yapx.utils.parse_sequence(
        None,
        "1",
        "two=2",
        "3",
        " four = 4 ",
        None,
        "five=",
    )
    assert args == [None, "1", "3", None]
    assert kwargs == {"two": "2", "four": "4", "five": None}
