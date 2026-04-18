import dataclasses
import operator
from collections.abc import Callable
from functools import partial
from math import exp, inf
from typing import Any, Literal

from .signature import Operator, Signature, make_checked_operator


def _cost_operator[V](
    unit: V,
    in_operator: Callable[[V, V], V],
    out_operator: Operator[V],
    out_to_in: Callable[[V], V],
    args: tuple[tuple[Any, bool], ...],
):
    result = unit
    outdomain = []

    for arg, arg_is_out in args:
        if arg_is_out:
            outdomain.append(arg)
        else:
            result = in_operator(result, arg)

    return in_operator(result, out_to_in(out_operator(*outdomain)))


def add_optimizer[S: Signature[Any]](
    signature: type[S],
    choose: Literal["min", "max"] = "min",
    **operators: Operator[float],
) -> S:
    """
    Create an optimization algebra for an additive cost.

    Note: The return type should be 'S[float]'. Unfortunately, Python’s type
    system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param signature: signature from which to derive the algebra
    :param choose: whether to do minimization or maximization
    :param operators: set of operators computing the additive terms
        and depending only on the outdomain arguments (default for
        missing operators: cost of 0)
    :returns: created algebra
    """
    elements: dict[str, Operator[float]] = {}

    for field in dataclasses.fields(signature):
        out_operator = operators.get(field.name, lambda *args: 0)
        op = partial(
            _cost_operator,
            0,
            operator.add,
            out_operator,
            lambda x: x,
        )
        elements[field.name] = make_checked_operator(field.type, int | float, op)

    if choose == "min":

        def null():
            return inf

        def choose_op(x, y):
            return min(x, y)

    else:

        def null():
            return -inf

        def choose_op(x, y):
            return max(x, y)

    elements["null"] = null
    elements["choose"] = choose_op
    return signature(**elements)


def boltzmann[S: Signature[Any]](
    signature: type[S],
    temperature: float,
    **operators: Operator[float],
) -> S:
    """
    Create an algebra computing Boltzmann weights based on an additive cost.

    Note: The return type should be 'S[float]'. Unfortunately, Python’s type
    system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param signature: signature from which to derive the algebra
    :param temperature: Boltzmann temperature, should be positive for
        minimization and negative for maximization; when the algebra is
        used for sampling, the sampling only retains optimal solutions as
        the temperature tends towards 0 and retains all solutions as it
        tends towards infinity
    :param operators: set of operators computing the additive terms
        and depending only on the outdomain arguments
    :returns: created algebra
    """
    elements: dict[str, Operator[float]] = {}

    for field in dataclasses.fields(signature):
        out_operator = operators.get(field.name, lambda *args: 0)
        op = partial(
            _cost_operator,
            1.0,
            operator.mul,
            out_operator,
            lambda x: exp(-(x / temperature)),
        )
        elements[field.name] = make_checked_operator(field.type, int | float, op)

    def null():
        return 0

    elements["null"] = null
    elements["choose"] = operator.add
    return signature(**elements)
