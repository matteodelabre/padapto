import dataclasses
from functools import partial
from typing import Any, TypeVar

from sowing import Node

from ..circuit import Circuit, make_node
from .signature import Signature, make_checked_operator


def _choose_operator(left: Circuit, right: Circuit) -> Circuit:
    # Flatten nested choice operators and remove nulls
    nodes = tuple(
        child_node
        for child_nodes in (
            child.children() if child.data.is_choose() else (child,)
            for child in (left, right)
        )
        for child_node in child_nodes
        if not child_node.data.is_null()
    )

    # Remove duplicate edges
    nodes = tuple(dict.fromkeys(nodes).keys())

    if len(nodes) == 0:
        return make_node(operator="null")
    elif len(nodes) == 1:
        return nodes[0]
    else:
        return make_node(operator="choose").extend(nodes)


def _trace_operator(
    name: str,
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
) -> Circuit:
    child_args: list[Circuit] = []
    out_args = []

    for arg_type, arg_value in zip(args_types, args, strict=True):
        if isinstance(arg_type, TypeVar):
            # Make null element absorbent in combinations
            if arg_value.data.is_null():
                return arg_value

            child_args.append(arg_value)
        else:
            out_args.append(arg_value)

    return make_node(operator=name, args=tuple(out_args)).extend(child_args)


def trace[S: Signature[Any]](signature: type[S]) -> S:
    """
    Derive an algebra that produces algebraic circuits from a signature.

    Each call to an algebra operator creates a node which is connected to its arguments,
    creating a circuit of solutions. The resulting circuits may be passed to the
    :func:`enumerate_solutions` and :func:`get_solution` to retrieve the solutions.

    Note: The return type should be 'S[Circuit]'. Unfortunately, Python’s type
    system is not powerful enough to express this yet (see
    <https://github.com/python/typing/issues/548>).

    :param signature: signature from which to derive the algebra
    :returns: created algebra
    """
    elements = {
        field.name: make_checked_operator(
            field.type,
            Node,
            partial(_trace_operator, field.name),
        )
        for field in dataclasses.fields(signature)
    }
    elements["choose"] = _choose_operator
    return signature(**elements)
