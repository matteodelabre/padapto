import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Any

from ..collections import Multiset
from .signature import (
    Operator,
    Signature,
    extract_algebra_parent,
    pipable,
    trace,
)


def _group_operator[T](
    operator: Operator[Multiset[T]],
    fields: tuple[str, ...],
    choose: Callable[[T, T], T],
    *args: Any,
) -> Multiset[T]:
    values = operator(*args)
    results = []
    keep = [True] * len(values)

    for i, value1 in enumerate(values):
        if not keep[i]:
            continue

        for j, value2 in enumerate(values[i + 1 :], start=i + 1):
            if all(
                getattr(value1, field) == getattr(value2, field) for field in fields
            ):
                value1 = choose(value1, value2)
                keep[j] = False

        results.append(value1)

    return Multiset(results)


@pipable
@trace(transparent=True)
def group[S: Signature[Multiset[Any]]](
    algebra: S,
    *fields: str,
) -> S:
    """
    Group values on sets of fields in the powerset of a joined algebra.

    After any operation in the algebra, the resulting multiset is traversed so that
    values that are equal on the given set of fields get grouped together. Grouped
    values get combined using the original choice function.

    The resulting algebra is valid if the given algebra is valid.

    :param algebra: power-joined algebra to transform
    :param fields: fields with respect to which to group
    :returns: new transformed algebra
    """
    if (joined := extract_algebra_parent(algebra, "power", index=0)) is None:
        raise TypeError("group: provided algebra is not a power algebra")

    if (subalgebras := extract_algebra_parent(joined, "join", kwargs=True)) is None:
        raise TypeError("group: provided algebra is not a joined algebra")

    for field in fields:
        if field not in subalgebras:
            raise TypeError(f"group: '{field}' is not a subalgebra field")

    signature = type(algebra)
    return signature(
        **{
            field.name: partial(
                _group_operator,
                getattr(algebra, field.name),
                fields,
                joined.choose,
            )
            for field in dataclasses.fields(signature)
        }
    )
