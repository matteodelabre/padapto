"""
An algebraic circuit is an encoding of a solution space as a DAG.

In most cases, this encoding allows representing an exponential number of solutions
inside a polynomial-sized object and to efficiently extract solutions and compute
summary information on the set of solutions.

See also :func:`padapto.algebras.trace` to create circuit-generating algebras from a
given signature.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from numbers import Real
from random import Random
from typing import TYPE_CHECKING, Any, TypeVar, get_args

from sowing import traversal
from sowing.node import Node
from sowing.repr import graphviz

if TYPE_CHECKING:
    from .algebras.signature import Signature


@dataclass(frozen=True)
class OperatorData:
    """Data attached to a node of an algebraic circuit."""

    # Name of the operator associated to the node
    operator: str

    # List of outdomain arguments for the operator with their argument index
    args: tuple[tuple[int, Any], ...] = ()

    def is_null(self) -> bool:
        """Test whether this is a null node."""
        return self.operator == "null"

    def is_choose(self) -> bool:
        """Test whether this is a choice node."""
        return self.operator == "choose"


type Circuit = Node[OperatorData, None]


def make_node(*args, **kwargs) -> Circuit:
    """Create an algebraic circuit node."""
    return Node(OperatorData(*args, **kwargs))


def default_circuit_style(data: OperatorData) -> graphviz.Style:
    """Render circuit nodes as GraphViz nodes."""
    if data.is_choose():
        return {
            "label": "⊕",
            "shape": "none",
            "width": "0",
            "height": "0",
        }
    else:
        head = data.operator
        args = data.args
        label = f"{head}({', '.join(map(repr, args))})" if args else head

        return {
            "shape": "box",
            "style": "rounded",
            "ordering": "out",  # preserve left-to-right order of outgoing edges
            "label": label,
        }


def render(
    circuit: Circuit,
    node_style: Callable[[OperatorData], graphviz.Style] = default_circuit_style,
    graph_style: graphviz.Style | None = None,
) -> str:
    """
    Represent a circuit in the DOT format.

    See :func:`sowing.repr.graphviz.write` for details.
    """
    if graph_style is None:
        graph_style = {}

    return graphviz.write(circuit, node_style=node_style, graph_style=graph_style)


def lazyproduct(*args: Iterable[Any]) -> Iterable[tuple]:
    """
    Lazy-evaluation equivalent to `itertools.product`.

    The standard `product` helper starts by fully evaluating its arguments and storing
    them in tuples, before yielding the first value. For long iterables, this causes a
    delay before the first result appears.
    """
    if not args:
        yield ()
    else:
        first = True
        saved = []

        for arg in args[0]:
            if first:
                for item in lazyproduct(*args[1:]):
                    saved.append(item)
                    yield (arg,) + item

                first = False
            else:
                for item in saved:
                    yield (arg,) + item


def enumerate_solutions(root: Circuit) -> Iterable[Circuit]:
    """
    Enumerate all solutions from a circuit.

    This function successively yields all the solutions represented by the circuit.
    While the number of solutions may be exponential in the size of the original
    circuit, this function only needs linear time and memory to go from a circuit
    to the next.

    :param root: root of the circuit to traverse
    :returns: iterator over all solutions
    """
    if root.data.is_null():
        return

    if root.data.is_choose():
        for child in root.children():
            yield from enumerate_solutions(child)

        return

    for children in lazyproduct(
        *(enumerate_solutions(child) for child in root.children())
    ):
        yield Node(root.data).extend(children)


def get_solution(circuit: Circuit) -> Circuit:
    """
    Extract a single arbitrary solution from a circuit.

    :param circuit: root of the circuit to traverse
    :returns: arbitrary solution
    """
    return next(iter(enumerate_solutions(circuit)))


def eval_inside[T](circuit: Circuit, alg: Signature[T]) -> dict[int, T]:
    """
    Evaluate a given algebra at each node of a circuit.

    :param circuit: circuit describing the structure to evaluate
    :param alg: algebra to use for evaluation
    :returns: dictionary associating each node ID to its value
    """
    signatures = {}

    for field in fields(alg):
        signatures[field.name] = get_args(field.type)[0]

    inside = {}

    for cursor in traversal.depth(circuit, preorder=False, unique="id"):
        node = cursor.node
        op_data = node.data

        children_values = [inside[id(child)] for child in node.children()]

        if op_data.is_choose():
            value = alg.multichoose(*children_values)
        else:
            # Collect and reorder children arguments and outdomain arguments
            signature = signatures[op_data.operator]
            args = [None] * len(signature)
            next_child = 0
            next_outdomain = 0

            for i, kind in enumerate(signature):
                if isinstance(kind, TypeVar):
                    args[i] = children_values[next_child]
                    next_child += 1
                else:
                    args[i] = op_data.args[next_outdomain]
                    next_outdomain += 1

            if len(children_values) != next_child:
                raise RuntimeError("number of children does not match signature")

            if len(op_data.args) != next_outdomain:
                raise RuntimeError(
                    "number of outdomain arguments does not match signature"
                )

            value = getattr(alg, op_data.operator)(*args)

        inside[id(node)] = value

    return inside


def eval_outside[T](
    circuit: Circuit, alg: Signature[T], inside: dict[int, Real]
) -> dict[int, Real]:
    """
    Evaluate the outside contributions of each node in a circuit.

    For each circuit node, its outside contribution is the total value of all solutions
    containing the node when treating this node as if it were a leaf.

    :param circuit: circuit describing the circuit to evaluate
    :param alg: algebra to use for evaluaiton
    :param inside: inside values as computed by :func:`eval_inside`
    :returns: dictionary associating each node ID to its outside value
    """
    outside = {}

    for cursor in traversal.depth(circuit, preorder=True, unique="id"):
        node = cursor.node

        if cursor.is_root():
            value = alg.unit()
        else:
            parent = cursor.up().node
            value = outside[id(parent)]

            if not parent.data.is_choose():
                for sibling in cursor.siblings():
                    # FIXME: Generalize to other kinds of products?
                    value *= inside[id(sibling.node)]

        outside[id(node)] = value

    return outside


def eval[T](circuit: Circuit, alg: Signature[T]) -> T:
    """
    Evaluate a circuit under an algebra.

    :param circuit: circuit describing the circuit to evaluate
    :param alg: algebra to use for evaluation
    :returns: value of the circuit as a whole
    """
    inside = eval_inside(circuit, alg)
    return inside[id(circuit)]


def sample(
    root: Circuit, gen: Random, weights: Signature[Real] | Mapping[int, Real]
) -> Circuit:
    """
    Randomly sample solutions from a circuit according to a specified weighting.

    :param root: circuit encoding the set of solutions to sample from
    :param gen: random source
    :param weights: weighting algebra for the solutions (use a standard counting algebra
        for uniform sampling), or a dictionary of weights computed from a weighting
        algebra through :func:`eval_inside`
    :returns: randomly sampled solution
    """
    if not isinstance(weights, Mapping):
        weights = eval_inside(root, weights)

    children = tuple(root.children())

    if root.data.is_choose():
        children_weights = [weights[id(child)] for child in children]
        index = gen.choices(range(len(children)), children_weights)[0]
        return sample(children[index], gen, weights)
    else:
        return Node(root.data).extend(sample(child, gen, weights) for child in children)
