import collections.abc
import shlex
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from .argparse_action import YapxAction, argparse_action
from .types import ArgumentParser, ArgValueType
from .utils import coalesce, is_instance, is_subclass

# @argparse_action
# # pylint: disable=unused-argument
# def to_str(
#     value: ArgValueType,
#     *,
#     action: argparse.Action,
#     parser: argparse.ArgumentParser,
#     namespace: argparse.Namespace,
#     option_string: Optional[str],
# ) -> Optional[str]:
#     return _cast(value, to=str)


T = TypeVar("T", bound=type)


def _split_csv_sequence(
    values: ArgValueType,
    target_type: T,
) -> Union[None, T, List[T]]:
    def _cast_type(txt: str, target_type: T) -> T:
        return (
            txt
            if is_instance(txt, target_type)
            else target_type[txt]
            if is_subclass(target_type, Enum)
            else target_type(txt)
        )

    if (
        values is None
        or not is_subclass(type(values), collections.abc.Sequence)
        or len(values) == 0
        or not isinstance(values[0], str)
    ):
        return values

    values_clean: str = (
        values.strip() if isinstance(values, str) else " ".join(list(values)).strip()
    )

    if values_clean and values_clean.startswith("[") and values_clean.endswith("]"):
        return [
            _cast_type(x, target_type=target_type)
            for x in shlex.split(values_clean.strip(" []"))
        ]

    return [_cast_type(x, target_type=target_type) for x in values]


def _get_target_type(action: YapxAction, parser: ArgumentParser) -> type:
    if not parser:
        return action.type
    # pylint: disable=protected-access
    return parser._inner_type_conversions.get(action.dest, action.type)


@argparse_action
# pylint: disable=unused-argument
def split_csv(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[List[Optional[Any]]]:
    return _split_csv_sequence(
        values,
        target_type=_get_target_type(action, parser),
    )


@argparse_action
def split_csv_to_tuple(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[Tuple[Optional[Any], ...]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values,
        target_type=_get_target_type(action, parser),
    )
    if split_values is not None:
        return tuple(split_values)
    return None


@argparse_action
def split_csv_to_set(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[Set[Optional[Any]]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values,
        target_type=_get_target_type(action, parser),
    )
    if split_values is not None:
        return set(split_values)
    return None


@argparse_action
def split_csv_to_dict(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[Dict[str, Optional[Any]]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values, target_type=str
    )

    if split_values is not None:
        return {
            x_split[0].strip(): None
            if len(x_split) < 2
            else coalesce(x_split[1].strip(), None, null_or_empty=True)
            for x in split_values
            if x
            for x_split in [x.split(parser.kv_separator, maxsplit=1)]
        }
    return None


@argparse_action
def str2enum(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[Enum]:
    if values is None:
        return None

    target_type: Type[Enum] = _get_target_type(action=action, parser=parser)
    if is_instance(values, target_type):
        return values

    return target_type[values]


@argparse_action(nargs=0)
def print_help(
    values: ArgValueType,
    *,
    parser: ArgumentParser,
    **kwargs: Any,
) -> None:
    parser.print_help()
    parser.exit()
