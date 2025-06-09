import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..collections import Multiset
from .join import join
from .power import power
from .signature import (
    Comparator,
    Operator,
    Signature,
    copy_algebra_metadata,
    get_algebra_metadata,
    make_natural_order,
    set_algebra_metadata,
)


def _make_pareto_wrapper[T](
    comparators: Mapping[str, Comparator[Any]],
    choose,
) -> Callable[[Operator[Multiset[T]]], Operator[Multiset[T]]]:
    def value_le(value1: T, value2: T) -> bool:
        return all(
            comparator(getattr(value1, name), getattr(value2, name))
            for name, comparator in comparators.items()
        )

    def value_eq(value1: T, value2: T) -> bool:
        return all(
            getattr(value1, name) == getattr(value2, name) for name in comparators
        )

    def pareto_filter(values: Iterable[T]) -> Multiset[T]:
        values = list(values)
        results = []
        keep = [True] * len(values)

        for i, value1 in enumerate(values):
            if not keep[i]:
                continue

            for j, value2 in enumerate(values):
                if i == j:
                    continue
                elif value_eq(value2, value1):
                    value1 = choose(value1, value2)
                    keep[j] = False
                elif value_le(value2, value1):
                    break
            else:
                results.append(value1)

        return Multiset(results)

    def pareto_wrapper(operator: Operator[Multiset[T]]) -> Operator[Multiset[T]]:
        def wrapped_operator(*args: Any) -> Multiset[T]:
            return pareto_filter(operator(*args))

        return wrapped_operator

    return pareto_wrapper


def pareto[S: Signature[Multiset[Any]]](
    algebra: S,
    *fields: str | tuple[str, Comparator[Any]],
) -> S:
    """
    Select non-dominated values in the powerset of a joined algebra.

    When choosing between two sets of values, given a set of fields from a joined
    algebra and a total monotonous order for each of them, keep all values for which
    there does not exist any one that is better or equal to them on all fields. If
    two values are equal on all fields, they get combined using the original choice
    function of the other subalgebras.

    The resulting algebra is valid if all the given comparator functions are total and
    monotonous orders.

    Note: Corresponds to the "Pareto product operator" as defined in "Pareto
    optimization in algebraic dynamic programming" by Saule and Giegerich (2015).
    """
    joined_algebra = get_algebra_metadata(algebra, power)
    subalgebras = get_algebra_metadata(algebra, join)

    if joined_algebra is None:
        raise TypeError("pareto: provided algebra is not a power algebra")

    if subalgebras is None:
        raise TypeError("pareto: provided algebra is not a joined algebra")

    comparators: dict[str, Comparator[Any]] = {}

    for field in fields:
        compare = None

        if not isinstance(field, str):
            field, compare = field

        if field not in subalgebras:
            raise TypeError(f"pareto: '{field}' is not a subalgebra field")

        if compare is None:
            comparators[field] = make_natural_order(subalgebras[field])
        else:
            comparators[field] = compare

    signature = type(algebra)
    pareto_wrapper = _make_pareto_wrapper(comparators, joined_algebra.choose)
    result = signature(
        **{
            field.name: pareto_wrapper(getattr(algebra, field.name))
            for field in dataclasses.fields(signature)
        }
    )
    copy_algebra_metadata(algebra, result)
    set_algebra_metadata(result, pareto, (algebra, fields))
    return result
