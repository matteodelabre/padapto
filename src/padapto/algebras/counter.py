import dataclasses
import operator
from typing import Any, TypeVar

from .signature import Operator, Signature, make_checked_operator


def _counter_operator(
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
) -> int:
    result = 1

    for arg_type, arg_value in zip(args_types, args, strict=True):
        if isinstance(arg_type, TypeVar):
            result *= arg_value

    return result


def counter(signature: type[Signature]) -> Signature[int]:
    """Create a counting algebra for the given signature."""
    elements: dict[str, Operator[int]] = {}

    for field in dataclasses.fields(signature):
        elements[field.name] = make_checked_operator(field.type, int, _counter_operator)

    elements["null"] = lambda: 0
    elements["choose"] = operator.add
    return signature(**elements)
