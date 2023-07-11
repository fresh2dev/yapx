import os
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Any, Dict, Optional, Tuple, Type

import pytest

import yapx
from yapx.arg import (
    ARGPARSE_ARG_METADATA_KEY,
    ArgparseArg,
    convert_to_command_string,
    convert_to_flag_string,
    convert_to_short_flag_string,
)


def test_convert_to_command_string() -> None:
    assert convert_to_command_string(" this_THAT ") == "this-that"
    assert convert_to_command_string(" _this_that ") == "that"
    assert convert_to_command_string(" x_that ") == "that"
    assert convert_to_command_string(" --WatIsThis ") == "wat-is-this"
    assert convert_to_command_string(" Wat Is This ") == "wat-is-this"


def test_convert_to_flag_string() -> None:
    assert convert_to_flag_string("--test") == "--test"
    assert convert_to_flag_string("test") == "--test"
    assert convert_to_flag_string("test_this") == "--test-this"
    assert convert_to_flag_string("t") == "-t"
    assert convert_to_flag_string("-t") == "-t"
    assert convert_to_flag_string(" --WatIsThis ") == "--wat-is-this"


def test_convert_to_short_flag_string() -> None:
    assert convert_to_short_flag_string("test") == "-t"
    assert convert_to_short_flag_string("  --TeST  ") == "-t"
    assert convert_to_short_flag_string("test_this") == "-t"
    assert convert_to_flag_string("t") == "-t"
    assert convert_to_flag_string("-t") == "-t"


def test_argument_parser_arg_defaults():
    # 1. ARRANGE
    expected: Dict[str, Any] = {
        "dest": "test_this",
        "option_strings": ["--test-this"],
        "type": None,
        "required": True,
        "group": None,
        "exclusive": False,
        "pos": False,
        "nargs": None,
        "const": None,
        "default": None,
        "choices": None,
        "help": None,
        "metavar": None,
        "action": None,
    }

    # 2. ACT
    arg: ArgparseArg = ArgparseArg(dest=expected["dest"])

    # 3. ASSERT
    assert arg
    assert is_dataclass(arg)
    assert arg.asdict() == expected


# pylint: disable=dangerous-default-value
def test_build_dataclass_from_func_raises_type_error():
    # 1. ARRANGE
    # pylint: disable=unused-argument,too-many-arguments
    def func1(value=lambda: []):
        pass

    def func2(value=lambda: {}):
        pass

    def func3(value=lambda: ()):
        pass

    def func4(value=b""):
        pass

    # 2,3. ACT & ASSERT
    for func in func1, func2, func3, func4:
        with pytest.raises(TypeError):
            yapx.utils.make_dataclass_from_func(func)


def _assert_model_field_attributes(
    model: Type[yapx.types.Dataclass],
    expected_field_props: Dict[str, Dict[str, Any]],
    expected_metadata_field_props: Dict[str, Dict[str, Any]],
):
    model_fields: Tuple[Field, ...] = fields(model)

    for f in model_fields:
        assert f.name in expected_field_props
        for exp_field_prop, exp_field_prop_value in expected_field_props[
            f.name
        ].items():
            assert getattr(f, exp_field_prop) == exp_field_prop_value

        assert ARGPARSE_ARG_METADATA_KEY in f.metadata
        f_metadata = f.metadata[ARGPARSE_ARG_METADATA_KEY]
        assert isinstance(f_metadata, ArgparseArg)

        assert f.name in expected_metadata_field_props
        assert f_metadata.asdict() == expected_metadata_field_props[f.name]


def test_build_dataclass_from_func_no_anno():
    # 1. ARRANGE
    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": str,
            "default": MISSING,
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg().asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(
        lambda value: ...,
    )

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )


def test_build_dataclass_from_func_none_anno():
    # 1. ARRANGE
    # pylint: disable=unused-argument,too-many-arguments
    def func(value=None):
        """testing docstring to argparse description"""
        ...

    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": Optional[str],
            "default": None,
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg(required=False).asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(func)

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )

    assert model.__doc__ == func.__doc__


def test_build_dataclass_from_func_default():
    # 1. ARRANGE
    # pylint: disable=unused-argument,too-many-arguments
    def func(value=3.14):
        pass

    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": float,
            "default": 3.14,
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg(required=False, default=3.14).asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(func)

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )


def test_build_dataclass_from_func_argfield_default():
    # 1. ARRANGE
    # pylint: disable=unused-argument,too-many-arguments
    def func(value=yapx.arg(default=3.14)):
        pass

    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": float,
            "default": 3.14,
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg(required=False, default=3.14).asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(func)

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )


def test_build_dataclass_from_func_argfield_envvar():
    # 1. ARRANGE
    env_var_name: str = "FUNKY_ARG"
    os.environ[env_var_name] = "3.1415"

    # pylint: disable=unused-argument,too-many-arguments
    def func(value=yapx.arg(env=env_var_name)):
        pass

    del os.environ[env_var_name]

    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": str,
            "default": "3.1415",
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg(required=False, default="3.1415").asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(func)

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )


def test_build_dataclass_from_func_argfield_envvar_file(clean_dir):
    # 1. ARRANGE
    env_var_name: str = "FUNKY_ARG"
    env_var_name_file: str = env_var_name + "_FILE"
    tmp_file: str = "test.tmp"
    os.environ[env_var_name_file] = tmp_file
    with open(tmp_file, "w+", encoding="utf8") as f:
        f.write("3.1415")

    # pylint: disable=unused-argument,too-many-arguments
    def func(value=yapx.arg(env=env_var_name)):
        pass

    del os.environ[env_var_name_file]
    os.remove(tmp_file)

    expected_field_props: Dict[str, Dict[str, Any]] = {
        "value": {
            "name": "value",
            "type": str,
            "default": "3.1415",
        },
    }
    expected_metadata_field_props: Dict[str, Dict[str, Any]] = {
        "value": ArgparseArg(required=False, default="3.1415").asdict(),
    }

    # 2. ACT
    model: Type[yapx.types.Dataclass] = yapx.utils.make_dataclass_from_func(func)

    # 3. ASSERT
    _assert_model_field_attributes(
        model,
        expected_field_props,
        expected_metadata_field_props,
    )
