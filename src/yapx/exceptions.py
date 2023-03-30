from typing import Any, Optional, Type


class ArgumentConflictError(Exception):
    ...


class ParserClosedError(Exception):
    ...


class NoArgsModelError(Exception):
    ...


class MutuallyExclusiveArgumentError(Exception):
    ...


class MutuallyExclusiveRequiredError(Exception):
    ...


class UnsupportedTypeError(TypeError):
    ...


def raise_unsupported_type_error(
    fld_type: Type[Any],
    from_exception: Optional[Exception] = None,
):
    fld_type_name: str = (
        fld_type.__name__ if hasattr(fld_type, "__name__") else str(fld_type)
    )

    msg: str = (
        f"Unsupported type: {fld_type_name}\nInstall 'pydantic' to support more types."
    )

    if from_exception:
        raise UnsupportedTypeError(msg) from from_exception
    raise UnsupportedTypeError(msg)
