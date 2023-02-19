from argparse import Action, ArgumentParser, Namespace
from functools import partial
from typing import Any, Callable, Optional, Type, TypeVar, Union

from .types import ArgValueType

__all__ = ["argparse_action"]


F = TypeVar("F", bound=Callable[..., Any])


class YapxAction(Action):
    def __init__(self, func: F, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._func = func
        self.name = func.__name__

    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: ArgValueType,
        option_string: Optional[str] = None,
    ) -> None:
        value = self._func(
            values,
            action=self,
            parser=parser,
            namespace=namespace,
            option_string=option_string,
        )
        setattr(namespace, self.dest, value)


def argparse_action(
    func: Optional[F] = None,
    **kwargs: Any,
) -> Union[Type[YapxAction], Callable[..., Type[YapxAction]]]:
    if not func:
        return partial(argparse_action, **kwargs)

    def _new_apx_action(**wrapper_kwargs: Any) -> Type[YapxAction]:
        class NewYapxAction(YapxAction):
            def __init__(self, **inner_kwargs: Any) -> None:
                inner_kwargs.update(wrapper_kwargs)
                super().__init__(func=func, **inner_kwargs)

        return NewYapxAction

    return _new_apx_action(**kwargs)
