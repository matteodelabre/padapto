import dataclasses
import itertools
import operator
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, cast

from ..collections import Multiset
from .signature import (
    Operator,
    Signature,
    copy_algebra_metadata,
    make_checked_operator,
    set_algebra_metadata,
)


def _power_operator[T, U](
    operator: Operator[T],
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
) -> Multiset[T]:
    return Multiset(
        operator(*call)
        for call in itertools.product(
            *(
                arg_value if isinstance(arg_type, TypeVar) else (arg_value,)
                for arg_type, arg_value in zip(args_types, args, strict=True)
            )
        )
    )


def power[S: Signature](algebra: S) -> S:
    """
    Take the power set of an algebra.

    Given an algebra on a type T, this function creates an algebra with the same
    signature valued over the power set of T. When choosing between two sets of values,
    the union of both is taken. When combining multiple sets of values, the original
    algebra function is called for each element of the Cartesian product of the sets.

    This algebra is valid if the original algebra is valid.

    Note: The return type should be 'S[Multiset[T]]', where 'T' is the type of the
    given algebra. Unfortunately, Python’s type system is not powerful enough to express
    this yet (see <https://github.com/python/typing/issues/548>).

    :param algebra: original algebra
    :returns: created algebra
    """
    signature = type(algebra)
    elements: dict[str, Operator[Multiset[Any]]] = {}

    for field in dataclasses.fields(signature):
        original = getattr(algebra, field.name)
        op = partial(_power_operator, original)
        elements[field.name] = make_checked_operator(field.type, Multiset, op)

    elements["null"] = cast(Callable[[], Multiset[Any]], lambda: Multiset())
    elements["choose"] = operator.add

    result = signature(**elements)
    copy_algebra_metadata(algebra, result)
    set_algebra_metadata(result, power, algebra)
    return result
