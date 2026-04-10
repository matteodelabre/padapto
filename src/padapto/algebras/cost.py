import dataclasses
import operator
from collections.abc import Callable
from functools import partial
from math import exp, inf
from numbers import Real
from typing import Any, Literal, TypeVar

from .signature import Operator, Signature, make_checked_operator


def _cost_operator[V: Real](
    unit: V,
    in_operator: Callable[[V, V], V],
    out_operator: Operator[V],
    out_to_in: Callable[[V], V],
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
):
    result = unit
    outdomain = []

    for arg_type, arg_value in zip(args_types, args, strict=True):
        if isinstance(arg_type, TypeVar):
            result = in_operator(result, arg_value)
        else:
            outdomain.append(arg_value)

    return in_operator(result, out_to_in(out_operator(*outdomain)))


def add_optimizer(
    signature: type[Signature],
    choose: Literal[min, max] = min,
    **operators: Operator[Real],
) -> Signature[Real]:
    """
    Create an optimization algebra for an additive cost.

    :param signature: signature for the algebra
    :param choose: whether to do minimization or maximization
    :param operators: set of operators computing the additive terms
        and depending only on the outdomain arguments
    """
    elements: dict[str, Operator[Real]] = {}

    for field in dataclasses.fields(signature):
        out_operator = operators.get(field.name, lambda *args: 0)
        op = partial(
            _cost_operator,
            0,
            operator.add,
            out_operator,
            lambda x: x,
        )
        elements[field.name] = make_checked_operator(field.type, Real, op)

    elements["null"] = (lambda: inf) if choose == min else (lambda: -inf)
    elements["choose"] = choose
    return signature(**elements)


def boltzmann(
    signature: type[Signature],
    temperature: Real,
    **operators: Operator[Real],
) -> Signature[Real]:
    """
    Create an algebra computing Boltzmann weights based on an additive cost.

    :param signature: signature for the algebra
    :param temperature: Boltzmann temperature, should be positive for
        minimization and negative for maximization; when the algebra is
        used for sampling, the sampling only retains optimal solutions as
        the temperature tends towards 0 and retains all solutions as it
        tends towards infinity
    :param operators: set of operators computing the additive terms
        and depending only on the outdomain arguments
    """
    elements: dict[str, Operator[Real]] = {}

    for field in dataclasses.fields(signature):
        out_operator = operators.get(field.name, lambda *args: 0)
        op = partial(
            _cost_operator,
            1,
            operator.mul,
            out_operator,
            lambda x: exp(-(x / temperature)),
        )
        elements[field.name] = make_checked_operator(field.type, Real, op)

    elements["null"] = lambda: 0
    elements["choose"] = operator.add
    return signature(**elements)
