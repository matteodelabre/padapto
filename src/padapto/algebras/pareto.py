import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..collections import Multiset
from .group import group
from .join import get_subalgebras, get_subrecord
from .signature import (
    Comparator,
    Operator,
    Signature,
    extract_algebra_parent,
    pipable,
    trace,
)


def _make_pareto_wrapper[T](
    comparators: Mapping[str, Comparator[Any]],
) -> Callable[[Operator[Multiset[T]]], Operator[Multiset[T]]]:
    def value_le(value1: T, value2: T) -> bool:
        return all(
            comparator(get_subrecord(value1, field), get_subrecord(value2, field))
            for field, comparator in comparators.items()
        )

    def value_eq(value1: T, value2: T) -> bool:
        return all(
            get_subrecord(value1, field) == get_subrecord(value2, field)
            for field in comparators
        )

    def pareto_filter(values: Iterable[T]) -> Multiset[T]:
        return Multiset(
            value
            for value in values
            if not any(
                value_le(other, value) and not value_eq(other, value)
                for other in values
            )
        )

    def pareto_wrapper(operator: Operator[Multiset[T]]) -> Operator[Multiset[T]]:
        def wrapped_operator(*args: Any) -> Multiset[T]:
            return pareto_filter(operator(*args))

        return wrapped_operator

    return pareto_wrapper


@pipable
@trace(transparent=True)
def pareto[S: Signature[Multiset[Any]]](algebra: S, *keys: str) -> S:
    """
    Select non-dominated values in the powerset of a joined algebra.

    When choosing between two sets of values, given a set of fields from a joined
    algebra, use the natural order of each field to keep all values for which there does
    not exist any one that is better or equal to them on all fields. If two values are
    equal on all fields, combine them using the original choice function.

    The resulting algebra is valid if the given algebra is valid and all the natural
    orders of the given fields are total and monotonous, which is the case if the choice
    functions of the respective subalgebras are conservative.

    Note: Corresponds to the "Pareto product operator" as defined in "Pareto
    optimization in algebraic dynamic programming" by Saule and Giegerich (2015).

    :param algebra: power-joined algebra to transform
    :param keys: fields with respect to which to select Pareto values; use dotted
        notation to access nested fields; use a star ('*') to access all fields at a
        given depth
    :returns: new transformed algebra
    """
    if (joined := extract_algebra_parent(algebra, "power", index=0)) is None:
        raise TypeError("pareto: provided algebra is not a power algebra")

    comparators: dict[str, Comparator[Any]] = {
        field: subalgebra.natural_order()
        for field, subalgebra in get_subalgebras(joined, *keys)
    }

    signature = type(algebra)
    pareto_wrapper = _make_pareto_wrapper(comparators)
    grouped = algebra | group(*comparators.keys())
    return signature(
        **{
            field.name: pareto_wrapper(getattr(grouped, field.name))
            for field in dataclasses.fields(signature)
        }
    )
