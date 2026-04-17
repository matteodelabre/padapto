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


def counter[S: Signature[Any]](signature: type[S]) -> S:
    """
    Create a counting algebra from a signature.

    Note: The return type should be 'S[int]'. Unfortunately, Python’s type
    system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param signature: signature from which to derive the algebra
    :returns: created algebra
    """
    elements: dict[str, Operator[int]] = {}

    for field in dataclasses.fields(signature):
        elements[field.name] = make_checked_operator(field.type, int, _counter_operator)

    def null():
        return 0

    elements["null"] = null
    elements["choose"] = operator.add
    return signature(**elements)
