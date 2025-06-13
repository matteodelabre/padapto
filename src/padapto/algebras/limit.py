import dataclasses
from typing import Any

from ..collections import Multiset
from .signature import (
    Operator,
    Signature,
    copy_algebra_metadata,
    get_algebra_metadata,
    pipable,
    set_algebra_metadata,
)


def _limit_wrap_operator[T](
    maxsize: int,
    operator: Operator[Multiset[T]],
) -> Operator[Multiset[T]]:
    def wrapped_operator(*args: Any) -> Multiset[T]:
        return Multiset(operator(*args)[:maxsize])

    return wrapped_operator


@pipable
def limit[S: Signature[Multiset[Any]]](algebra: S, maxsize: int) -> S:
    """
    Limit the number of entries in each value of a power algebra.

    This algebra is only valid if the original algebra is valid and if :param:`maxsize`
    is zero, one or infinity. For other values of :param:`maxsize`, it is guaranteed
    that the resulting multisets are subsets of the complete set of results, whose size
    is upper bounded by the parameter.

    :param alg: original algebra
    :param maxsize: maximum size of sets to yield
    :returns: limited algebra
    """
    signature = type(algebra)
    result = signature(
        **{
            field.name: _limit_wrap_operator(maxsize, getattr(algebra, field.name))
            for field in dataclasses.fields(signature)
        }
    )

    copy_algebra_metadata(algebra, result)

    if (og_limit := get_algebra_metadata(result, limit)) is not None:
        og_algebra, og_maxsize = og_limit
        set_algebra_metadata(result, limit, (og_algebra, min(maxsize, og_maxsize)))
    else:
        set_algebra_metadata(result, limit, (algebra, maxsize))

    return result
