import dataclasses
from collections.abc import Callable
from typing import Any

from .signature import (
    Comparator,
    Signature,
    extract_algebra_parent,
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
        left_value = getattr(left, field)
        right_value = getattr(right, field)

        left_le_right = compare(left_value, right_value)
        right_le_left = compare(right_value, left_value)

        if left_le_right and right_le_left:
            return prev_choose(left, right)

        if left_le_right:
            return left

        if right_le_left:
            return right

        raise TypeError(f"lex: order for field '{field}' is not total")

    return choose


@pipable
@trace(transparent=True)
def lex[S: Signature[Any]](algebra: S, *fields: str | tuple[str, Comparator[Any]]) -> S:
    """
    Select between values based on a lexicographical order in a joined algebra.

    When choosing between two values, given a set of fields from a joined algebra and a
    total monotonous order for each of them, keep the least one in lexicographical order
    (i.e., first choose the least value according to the first field, then compare the
    second field in case of equality, and so on). If two values are equal on all fields,
    they get combined using the original choice function of the other subalgebras.

    The resulting algebra is valid if all the given comparator functions are total and
    monotonous orders.

    Repeated call of this function on multiple fields is equivalent to a single call on
    all the fields, i.e., `lex(alg, "f1", "f2", "f3")` is equivalent to
    `lex(lex(lex(alg, "f3"), "f2"), "f1")`

    Note: Corresponds to the "Product operation on evaluation algebras" when defined
    on singly-valued algebras, as defined in "Versatile and declarative dynamic
    programming using pair algebras" by Steffen and Giegerich (2005).

    :param algebra: joined algebra to transform
    :param fields: fields with respect to which to order - each argument is either a
        tuple containing a field name and a comparator giving the total order to use for
        that field, or simply a plain field name (in which case the natural order of the
        subalgebra is used, which is always total and monotonous if the choice function
        of that subalgebra is conservative)
    :returns: new transformed algebra
    """
    if (subalgebras := extract_algebra_parent(algebra, "join", kwargs=True)) is None:
        raise TypeError("lex: provided algebra is not a joined algebra")

    choose = algebra.choose

    for field in reversed(fields):
        compare = None

        if not isinstance(field, str):
            field, compare = field

        if field not in subalgebras:
            raise TypeError(f"lex: '{field}' is not a subalgebra field")

        if compare is None:
            compare = make_natural_order(subalgebras[field])

        choose = _make_lex_choice(choose, field, compare)

    return dataclasses.replace(algebra, choose=choose)
