import dataclasses
from typing import Any

from ..collections import Multiset
from .signature import (
    Operator,
    Signature,
    pipable,
    trace,
)


def _limit_wrap_operator[T](
    maxsize: int,
    operator: Operator[Multiset[T]],
) -> Operator[Multiset[T]]:
    def wrapped_operator(*args: Any) -> Multiset[T]:
        return Multiset(operator(*args)[:maxsize])

    return wrapped_operator


@pipable
@trace(transparent=True)
def limit[S: Signature[Multiset[Any]]](algebra: S, maxsize: int | None) -> S:
    """
    Limit the number of entries in each value of a power algebra.

    This algebra is only valid if the original algebra is valid and if :param:`maxsize`
    is zero, one or infinity. For other values of :param:`maxsize`, it is guaranteed
    that the resulting multisets are subsets of the complete set of results, whose size
    is upper bounded by the parameter.

    :param alg: original algebra
    :param maxsize: maximum size of sets to yield, or None to disable limiting
    :returns: limited algebra
    """
    if maxsize is None:
        return algebra

    signature = type(algebra)
    return signature(
        **{
            field.name: _limit_wrap_operator(maxsize, getattr(algebra, field.name))
            for field in dataclasses.fields(signature)
        }
    )
