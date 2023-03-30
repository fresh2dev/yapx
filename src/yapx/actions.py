import collections.abc
import shlex
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar

from .argparse_action import YapxAction, argparse_action
from .types import ArgumentParser, ArgValueType
from .utils import coalesce, try_isinstance, try_issubclass

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
) -> Optional[List[T]]:
    if isinstance(values, str) and values:
        values = [values]
    elif (
        values is None
        or not try_issubclass(type(values), collections.abc.Sequence)
        or len(values) == 0
        or not isinstance(values[0], str)
    ):
        return values

    all_values: List[str] = []

    list_prefix: str = "list["
    list_suffix: str = "]"

    if all(x.endswith(",") for x in values[:-1]):
        values = [x.rstrip(",") for x in values]

    for value in values:
        value_clean: str = value.strip()

        if (
            value_clean
            and value_clean.lower().startswith(list_prefix)
            and value_clean.lower().endswith(list_suffix)
        ):
            all_values.extend(
                _split_csv_sequence(
                    values=shlex.split(
                        value_clean[len(list_prefix) : -len(list_suffix)].strip(),
                    ),
                    target_type=target_type,
                ),
            )
        else:
            all_values.append(target_type(value))

    return all_values


@argparse_action
def split_csv(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **_kwargs: Any,
) -> Optional[List[Optional[Any]]]:
    return _split_csv_sequence(
        values,
        target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
        ),
    )


@argparse_action
def split_csv_to_tuple(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **_kwargs: Any,
) -> Optional[Tuple[Optional[Any], ...]]:
    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values,
        target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
        ),
    )

    if split_values is None:
        return None

    return tuple(split_values)


@argparse_action
def split_csv_to_set(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **_kwargs: Any,
) -> Optional[Set[Optional[Any]]]:
    if try_isinstance(values, set):
        return values

    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values,
        target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
        ),
    )
    if split_values is None:
        return None

    return set(split_values)


def _split_csv_to_dict(
    values: ArgValueType,
    kv_separator: str,
    target_value_type: Optional[Type[Any]] = None,
) -> Optional[Dict[str, Optional[Any]]]:
    if try_isinstance(values, dict):
        return values

    split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
        values,
        target_type=str,
    )

    if split_values is None:
        return None

    return {
        x_split[0].strip(): (
            v if v is None or target_value_type is None else target_value_type(v)
        )
        for x in split_values
        if x
        for x_split in [x.split(kv_separator, maxsplit=1)]
        for v in [
            (
                None
                if len(x_split) < 2
                else coalesce(x_split[1].strip(), None, null_or_empty=True)
            ),
        ]
    }


@argparse_action
def split_csv_to_dict(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    **_kwargs: Any,
) -> Optional[Dict[str, Optional[Any]]]:
    return _split_csv_to_dict(
        values,
        kv_separator=parser.kv_separator,
        target_value_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
            str,
        ),
    )
