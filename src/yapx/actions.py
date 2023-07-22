import collections.abc
import shlex
from argparse import Action, Namespace, _AppendAction, _CountAction
from argparse import _HelpAction as HelpAction
from copy import copy
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from .types import ArgumentParser, ArgValueType
from .utils import coalesce, try_isinstance


class BooleanOptionalAction(Action):
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
        self._negation_option_strings: List[str] = []

        _option_strings = []
        for option_string in option_strings:
            if not option_string.startswith("--"):
                _option_strings.append(option_string)
            elif "/" in option_string:
                opt_str, opt_str_neg = map(
                    str.strip,
                    option_string.split("/", maxsplit=1),
                )

                _option_strings.extend([opt_str, opt_str_neg])

                self._negation_option_strings.append(opt_str_neg)
            elif not option_string.startswith(self.NEGATION_PREFIX):
                opt_str_neg = self.NEGATION_PREFIX + option_string[2:]
                _option_strings.extend(
                    [
                        option_string,
                        opt_str_neg,
                    ],
                )
                self._negation_option_strings.append(opt_str_neg)
            else:  # if option_string.startswith(self.NEGATION_PREFIX):
                opt_str_neg = "--" + option_string[len(self.NEGATION_PREFIX) :]
                if default is False:
                    _option_strings.extend(
                        [
                            opt_str_neg,
                            option_string,
                        ],
                    )
                else:  # elif default is None or default:
                    _option_strings.extend(
                        [
                            option_string,
                            opt_str_neg,
                        ],
                    )

                self._negation_option_strings.append(opt_str_neg)

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
            value: bool = option_string not in self._negation_option_strings
        else:
            dest_type: Type[Any] = parser._dest_type.get(self.dest, bool)
            value = dest_type(values)

        setattr(
            namespace,
            self.dest,
            value,
        )

    def format_usage(self):
        return " | ".join(self.option_strings)


class HelpAllAction(HelpAction):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: ArgValueType,
        option_string: Optional[str] = None,
    ):
        parser.print_help(include_commands=True)
        parser.exit()


class CountAction(_CountAction):
    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        required=False,
        help=None,  # pylint: disable=redefined-builtin
        **_kwargs,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        count = getattr(namespace, self.dest, None)
        if count is None:
            count = 0

        if option_string:
            count += 1

        setattr(namespace, self.dest, count)


class FeatureFlagAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if option_string:
            values = option_string
        setattr(namespace, self.dest, values.lstrip("-"))


T = TypeVar("T", bound=type)


def _append_new_dest_values(
    namespace: Namespace,
    dest: str,
    new_values: Sequence[T],
) -> Sequence[T]:
    # ref: https://github.com/python/cpython/blob/main/Lib/argparse.py#L791

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
        or not try_isinstance(values, collections.abc.Sequence)
        or len(values) == 0
        or not isinstance(values[0], str)
    ):
        return values

    all_values: List[str] = []

    list_prefix: str = "["
    list_suffix: str = "]"

    # if all (ignoring last) end with comma, remove comma.
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


# https://github.com/python/cpython/blob/main/Lib/argparse.py#L140C23-L140C23
def _copy_items(items: Optional[List[T]]) -> List[T]:
    if items is None:
        return []
    # The copy module is used only in the 'append' and 'append_const'
    # actions, and it is needed only when the default value isn't a list.
    # Delay its import for speeding up the common case.
    if type(items) is list:
        return items[:]

    return copy(items)


class SplitCsvListAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        split_values: Optional[List[str]] = _split_csv_sequence(
            values,
            target_type=parser._dest_type.get(  # pylint: disable=protected-access
                self.dest,
            ),
        )
        items = _append_new_dest_values(
            namespace,
            dest=self.dest,
            new_values=split_values,
        )

        setattr(namespace, self.dest, items)


class SplitCsvTupleAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        split_values: Optional[List[str]] = _split_csv_sequence(
            values,
            target_type=parser._dest_type.get(  # pylint: disable=protected-access
                self.dest,
            ),
        )
        items = _append_new_dest_values(
            namespace,
            dest=self.dest,
            new_values=split_values,
        )

        setattr(namespace, self.dest, tuple(items) if items is not None else None)


class SplitCsvSetAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        split_values: Optional[List[str]] = _split_csv_sequence(
            values,
            target_type=parser._dest_type.get(  # pylint: disable=protected-access
                self.dest,
            ),
        )
        items = _append_new_dest_values(
            namespace,
            dest=self.dest,
            new_values=split_values,
        )

        setattr(namespace, self.dest, set(items) if items is not None else None)


class SplitCsvDictAction(_AppendAction):
    def __call__(self, parser, namespace, values, option_string=None):
        split_values_dict: Optional[Dict[str, Any]] = None

        if not values or isinstance(values, dict):
            split_values_dict = values
        else:
            split_values: Optional[List[str]] = _split_csv_sequence(
                values,
                target_type=str,
            )
            if split_values:
                split_values_dict = {
                    x_split[0].strip(): v
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

        # pylint: disable=protected-access
        if split_values_dict and self.dest in parser._dest_type:
            target_type: Type[Any] = parser._dest_type[self.dest]
            if target_type:
                split_values_dict = {
                    k: target_type(v) for k, v in split_values_dict.items()
                }

        items = _append_new_dest_values(
            namespace,
            dest=self.dest,
            new_values=split_values_dict,
        )

        setattr(namespace, self.dest, items)
