from dataclasses import dataclass
from typing import Optional

import pytest

import yapx


@pytest.mark.parametrize(
    "x",
    [-1, 1, "1", "true", "t", "yes", "y", "on", True, ["hello"]],
)
def test_cast_bool_true(x):
    assert yapx.utils.cast_bool(x) is True


@pytest.mark.parametrize("x", [0, "0", "false", "f", "no", "n", "off", None, False, []])
def test_cast_bool_false(x):
    assert yapx.utils.cast_bool(x) is False


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
