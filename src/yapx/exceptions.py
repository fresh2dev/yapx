from argparse import ArgumentTypeError
from typing import Any, Optional, Type


class NoArgsModelError(Exception):
    ...


class UnsupportedTypeError(ArgumentTypeError):
    ...


def raise_unsupported_type_error(
    fld_type: Type[Any],
    from_exception: Optional[Exception] = None,
):
    fld_type_name: str = (
        fld_type.__name__ if hasattr(fld_type, "__name__") else str(fld_type)
    )

    msg: str = f"Unsupported type: {fld_type_name}\n\"pip install 'yapx[pydantic]'\" to support more types."

    if from_exception:
        raise UnsupportedTypeError(msg) from from_exception
    raise UnsupportedTypeError(msg)
