import dataclasses
from collections.abc import Callable
from typing import Any

from .join import get_subalgebras, get_subrecord
from .signature import (
    Comparator,
    Signature,
    make_natural_order,
    pipable,
    trace,
)


def _make_lex_choice[T](
    prev_choose: Callable[[T, T], T],
    field: str,
    compare: Comparator[Any],
) -> Callable[[T, T], T]:
    def choose(left: T, right: T) -> T:
        left_value = get_subrecord(left, field)
        right_value = get_subrecord(right, field)

        left_le_right = compare(left_value, right_value)
        right_le_left = compare(right_value, left_value)

        if left_le_right and not right_le_left:
            return left

        if right_le_left and not left_le_right:
            return right

        return prev_choose(left, right)

    return choose


@pipable
@trace(transparent=True)
def lex[S: Signature[Any]](algebra: S, *keys: str) -> S:
    """
    Select between values based on a lexicographical order in a joined algebra.

    When choosing between two values, given a set of fields from a joined algebra, use
    the natural order of each field to keep the least value in lexicographical order
    (i.e., first choose the least value according to the first field’s order, then
    compare the second field in case of equality, and so on). If two values are equal on
    all fields, combine them using the original choice function.

    The resulting algebra is valid if all the natural orders of all given fields are
    total and monotonous, which is the case if the choice functions of the respective
    subalgebras are conservative.

    Repeated call of this function on multiple fields is equivalent to a single call on
    all the fields, i.e., `lex(alg, "f1", "f2", "f3")` is equivalent to
    `lex(lex(lex(alg, "f3"), "f2"), "f1")`

    Note: Corresponds to the "Product operation on evaluation algebras" when defined
    on singly-valued algebras, as defined in "Versatile and declarative dynamic
    programming using pair algebras" by Steffen and Giegerich (2005).

    :param algebra: joined algebra to transform
    :param keys: fields with respect to which to order; use dotted notation to
        access nested fields; use a star ('*') to access all fields at a given depth
    :returns: new transformed algebra
    """
    choose = algebra.choose

    for field, subalgebra in list(get_subalgebras(algebra, *keys))[::-1]:
        compare = make_natural_order(subalgebra)
        choose = _make_lex_choice(choose, field, compare)

    return dataclasses.replace(algebra, choose=choose)
