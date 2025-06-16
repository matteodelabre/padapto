import dataclasses
import itertools
import operator
from collections.abc import Callable
from functools import partial, total_ordering
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

from ..collections import Multiset
from .signature import (
    Comparator,
    Operator,
    Signature,
    copy_algebra_metadata,
    make_checked_operator,
    make_natural_order,
    pipable,
    set_algebra_metadata,
)


def _merge_multisets[T](
    compare: Comparator[T],
    left: Multiset[T],
    right: Multiset[T],
) -> Multiset[T]:
    """Merge two multisets in sorted order."""
    result: list[T | None] = [None] * (len(left) + len(right))
    left_idx = 0
    right_idx = 0

    for place in range(len(result)):
        if right_idx == len(right):
            result[place] = left[left_idx]
            left_idx += 1
            continue

        if left_idx == len(left):
            result[place] = right[right_idx]
            right_idx += 1
            continue

        left_item = left[left_idx]
        right_item = right[right_idx]

        if compare(right_item, left_item) and not compare(left_item, right_item):
            result[place] = right_item
            right_idx += 1
        else:
            result[place] = left_item
            left_idx += 1

    return Multiset(cast(list[T], result))


def _compare_to_key[T](
    compare: Comparator[T],
) -> Callable[[T], "SupportsRichComparison"]:
    @total_ordering
    class Key:
        def __init__(self, item: T):
            self.item = item

        def __lt__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented

            return compare(self.item, other.item) and not compare(other.item, self.item)

        def __eq__(self, other: Any) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented

            return compare(self.item, other.item) and compare(other.item, self.item)

    return Key


def _power_operator[T, U](
    operator: Operator[T],
    compare: Comparator[T] | None,
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
) -> Multiset[T]:
    result = [
        operator(*call)
        for call in itertools.product(
            *(
                arg_value if isinstance(arg_type, TypeVar) else (arg_value,)
                for arg_type, arg_value in zip(args_types, args, strict=True)
            )
        )
    ]

    if compare is not None:
        result.sort(key=_compare_to_key(compare))

    return Multiset(result)


@pipable
def power[S: Signature[Any]](
    algebra: S, order: Comparator[Any] | bool = False
) -> S:
    """
    Take the power set of an algebra.

    Given an algebra on a type T, this function creates an algebra with the same
    signature valued over sets of T. When choosing between two sets of values, the union
    of both is taken. When combining multiple sets of values, the original algebra
    function is called for each element of the Cartesian product of the sets.

    If an order is provided, the multisets elements are ordered according to this order.

    This algebra is valid if the original algebra is valid.

    Note: The return type should be `S[Multiset[T]]`, where `T` is the type of the given
    algebra. Similarly, the type of :param:`order` should be `Comparator[T]`.
    Unfortunately, Python’s type system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param algebra: original algebra
    :param order: sort the set elements according to this order (if True, use the
        natural ordering of the original algebra; if False, do not sort the elements;
        the default is False)
    :returns: created algebra
    """
    if order is True:
        compare = make_natural_order(algebra)
    elif order is False:
        compare = None
    else:
        compare = order

    signature = type(algebra)
    elements: dict[str, Operator[Multiset[Any]]] = {}

    for field in dataclasses.fields(signature):
        original = getattr(algebra, field.name)
        op = partial(_power_operator, original, compare)
        elements[field.name] = make_checked_operator(field.type, Multiset, op)

    elements["null"] = cast(Callable[[], Multiset[Any]], lambda: Multiset())

    if compare is not None:
        elements["choose"] = partial(_merge_multisets, compare)
    else:
        elements["choose"] = operator.add

    result = signature(**elements)
    copy_algebra_metadata(algebra, result)
    set_algebra_metadata(result, power, algebra)
    return result
