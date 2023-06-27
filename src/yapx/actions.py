import collections.abc
import shlex
from argparse import Action, Namespace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

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


class YapxBooleanAction(Action):
    # ref: https://github.com/python/cpython/blob/main/Lib/argparse.py#L889C1-L943C47
    NEGATION_PREFIX: str = "--no-"

    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,  # pylint: disable=redefined-builtin
        choices=None,
        required=False,
        help=None,  # pylint: disable=redefined-builtin
        metavar=None,
    ):
        self._base_case_is_negative: bool = False
        self._negative_option_strings: List[str] = []

        if option_strings:
            self._base_case_is_negative = any(
                x.startswith(self.NEGATION_PREFIX) for x in option_strings
            )

        if self._base_case_is_negative and default is not None:
            default = not default

        _option_strings = []
        for option_string in option_strings:
            if "/" in option_string:
                opt_str, opt_str_neg = map(
                    str.strip,
                    option_string.split("/", maxsplit=1),
                )

                _option_strings.extend([opt_str, opt_str_neg])
                self._negative_option_strings.append(opt_str_neg)
            elif not option_string.startswith("--"):
                _option_strings.append(option_string)
            elif not option_string.startswith(self.NEGATION_PREFIX):
                _option_strings.extend(
                    [
                        option_string,
                        self.NEGATION_PREFIX + option_string[2:],
                    ],
                )
            elif default is False:
                _option_strings.extend(
                    [
                        "--" + option_string[len(self.NEGATION_PREFIX) :],
                        option_string,
                    ],
                )
            else:  # elif default is None or default:
                _option_strings.extend(
                    [
                        option_string,
                        "--" + option_string[len(self.NEGATION_PREFIX) :],
                    ],
                )

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string:
            value: bool = (
                not option_string.startswith(self.NEGATION_PREFIX)
                and option_string not in self._negative_option_strings
            )
        else:
            value = values

        if self._base_case_is_negative and value is not None:
            value = not value

        setattr(
            namespace,
            self.dest,
            value,
        )

    def format_usage(self):
        return " | ".join(self.option_strings)


@argparse_action
def counting_action(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    namespace: Namespace,
    **_kwargs: Any,
) -> int:
    value: Optional[int] = getattr(namespace, action.dest, None)
    return 0 if value is None else value + 1


@argparse_action
def featflag_action(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    namespace: Namespace,
    option_string: Optional[str] = None,
    **_kwargs: Any,
) -> Optional[str]:
    if option_string:
        values = option_string
    return values.lstrip("-")


T = TypeVar("T", bound=type)


def _append_new_dest_values(
    namespace: Namespace,
    dest: str,
    new_values: Sequence[T],
    # ref: https://github.com/python/cpython/blob/main/Lib/argparse.py#L791
) -> Sequence[T]:
    items = getattr(namespace, dest, None)

    if new_values is not None:
        if not items:
            items = new_values
        elif isinstance(new_values, dict):
            items = dict(items)
            items.update(new_values)
        else:
            items = list(items)
            items.extend(new_values)

    return items


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

    list_prefix: str = "["
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
    namespace: Namespace,
    **_kwargs: Any,
) -> Optional[List[Optional[Any]]]:
    split_values: Optional[List[str]] = _split_csv_sequence(
        values,
        target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
        ),
    )
    return _append_new_dest_values(namespace, dest=action.dest, new_values=split_values)


@argparse_action
def split_csv_to_tuple(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    namespace: Namespace,
    **_kwargs: Any,
) -> Optional[Tuple[Optional[Any], ...]]:
    split_values: Optional[List[Optional[Any]]] = (
        values
        if isinstance(values, tuple)
        else _split_csv_sequence(
            values,
            target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
                action.dest,
            ),
        )
    )
    new_value = _append_new_dest_values(
        namespace,
        dest=action.dest,
        new_values=split_values,
    )

    return tuple(new_value) if new_value is not None else None


@argparse_action
def split_csv_to_set(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    namespace: Namespace,
    **_kwargs: Any,
) -> Optional[Set[Optional[Any]]]:
    split_values: Optional[List[Optional[Any]]] = (
        values
        if isinstance(values, set)
        else _split_csv_sequence(
            values,
            target_type=parser._inner_type_conversions.get(  # pylint: disable=protected-access
                action.dest,
            ),
        )
    )
    new_value = _append_new_dest_values(
        namespace,
        dest=action.dest,
        new_values=split_values,
    )

    return set(new_value) if new_value is not None else None


@argparse_action
def split_csv_to_dict(
    values: ArgValueType,
    *,
    action: YapxAction,
    parser: ArgumentParser,
    namespace: Namespace,
    **_kwargs: Any,
) -> Optional[Dict[str, Optional[Any]]]:
    target_value_type = (
        parser._inner_type_conversions.get(  # pylint: disable=protected-access
            action.dest,
            str,
        )
    )

    new_values: Optional[Dict[str, Any]] = None

    if try_isinstance(values, dict):
        new_values = values
    else:
        split_values: Optional[List[Optional[Any]]] = _split_csv_sequence(
            values,
            target_type=str,
        )

        if split_values is not None:
            new_values = {
                x_split[0].strip(): (
                    v
                    if v is None or target_value_type is None
                    else target_value_type(v)
                )
                for x in split_values
                if x
                for x_split in [x.split(parser.kv_separator, maxsplit=1)]
                for v in [
                    (
                        None
                        if len(x_split) < 2
                        else coalesce(x_split[1].strip(), None, null_or_empty=True)
                    ),
                ]
            }

    return _append_new_dest_values(namespace, action.dest, new_values=new_values)
