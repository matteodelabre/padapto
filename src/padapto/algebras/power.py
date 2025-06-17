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
    make_checked_operator,
    make_natural_order,
    pipable,
    trace,
)


def _merge_multisets[T](
    compare: Comparator[T],
    unique: bool,
    left: Multiset[T],
    right: Multiset[T],
) -> Multiset[T]:
    """
    Merge two multisets in sorted order.

    :param compare: binary comparator function indicating whether lhs <= rhs
    :param unique: if true, duplicates are removed
    :param left: first multiset to merge
    :param right: second multiset to merge
    :return: merged values in sorted order
    """
    result: list[T | None] = [None] * (len(left) + len(right))
    place = 0
    left_idx = 0
    right_idx = 0

    while left_idx < len(left) or right_idx < len(right):
        if right_idx == len(right):
            result[place] = left[left_idx]
            left_idx += 1
        elif left_idx == len(left):
            result[place] = right[right_idx]
            right_idx += 1
        else:
            left_item = left[left_idx]
            right_item = right[right_idx]

            left_le_right = compare(left_item, right_item)
            right_le_left = compare(right_item, left_item)

            if left_le_right and right_le_left and unique:
                result[place] = left_item
                left_idx += 1
                right_idx += 1
            elif right_le_left and not left_le_right:
                result[place] = right_item
                right_idx += 1
            else:
                result[place] = left_item
                left_idx += 1

        place += 1

    return Multiset(cast(list[T], result[:place]))


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
    unique: bool,
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

    if unique:
        result = list(dict.fromkeys(result))

    return Multiset(result)


@pipable
@trace()
def power[S: Signature[Any]](
    algebra: S, *, order: Comparator[Any] | bool = False, unique: bool = False
) -> S:
    """
    Take the power set of an algebra.

    Given an algebra on a type T, this function creates an algebra with the same
    signature valued over multisets of T. When choosing between two sets of values, the
    union of both is taken. When combining multiple sets of values, the original algebra
    function is called for each element of the Cartesian product of the multisets.

    If an order is provided, the multisets elements are ordered according to this order
    (multisets have an intrinsic order, but two multisets containing the same elements
    in a different order are considered to be equal).

    This algebra is valid if the original algebra is valid.

    Note: The return type should be `S[Multiset[T]]`, where `T` is the type of the given
    algebra. Similarly, the type of :param:`order` should be `Comparator[T]`.
    Unfortunately, Python’s type system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param algebra: original algebra
    :param order: sort the multiset elements according to this order (if True, use the
        natural ordering of the original algebra; if False, do not sort the elements;
        the default is False)
    :param unique: remove the duplicate values from the multiset (the default is to keep
        duplicates)
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
        op = partial(_power_operator, original, compare, unique)
        elements[field.name] = make_checked_operator(field.type, Multiset, op)

    def null() -> Multiset[Any]:
        return Multiset()

    elements["null"] = null

    if compare is not None:
        elements["choose"] = partial(_merge_multisets, compare, unique)
    elif unique:

        def choose(left: Multiset[Any], right: Multiset[Any]) -> Multiset[Any]:
            return Multiset(dict.fromkeys(left + right))

        elements["choose"] = choose
    else:
        elements["choose"] = operator.add

    return signature(**elements)
