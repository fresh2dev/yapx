import collections.abc
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar

from .argparse_action import YapxAction, argparse_action
from .types import ArgumentParser, ArgValueType

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


def _split_csv_sequence(
    values: ArgValueType,
    cast_to: type = str,
) -> Optional[List[Optional[Any]]]:

    T = TypeVar("T", bound=type)

    def _split_csv_str(txt: str, cast_to: T) -> List[Optional[T]]:
        return [cast_to(y) for x in txt.split(",") for y in [x.strip()] if y]

    if values is None:
        return None

    if isinstance(values, str):
        return _split_csv_str(values, cast_to=cast_to)

    if values and issubclass(type(values), collections.abc.Sequence):
        return [y for x in values for y in _split_csv_str(x, cast_to=cast_to)]

    return []


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
    return _split_csv_sequence(values, cast_to=_get_target_type(action, parser))


@argparse_action
def split_csv_to_tuple(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **kwargs: Any,
) -> Optional[Tuple[Optional[Any], ...]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values, cast_to=_get_target_type(action, parser)
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
        values, cast_to=_get_target_type(action, parser)
    )
    if split_values is not None:
        return set(split_values)
    return None


@argparse_action
def split_csv_to_dict(
    values: ArgValueType,
    **kwargs: Any,
) -> Optional[Dict[str, Optional[Any]]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(values)

    if split_values is not None:
        return {
            x_split[0].strip(): None if len(x_split) < 2 else x_split[1].strip()
            for x in split_values
            if x
            for x_split in [x.split(":" if ":" in x else "=", maxsplit=1)]
        }
    return None


@argparse_action(nargs=0)
def print_help(
    values: ArgValueType,
    *,
    parser: ArgumentParser,
    **kwargs: Any,
) -> None:
    parser.print_help()
    parser.exit()
