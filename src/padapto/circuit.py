"""
An algebraic circuit is an encoding of a solution space as a DAG.

In most cases, this encoding allows representing an exponential number of solutions
inside a polynomial-sized object and to efficiently extract solutions and compute
summary information on the set of solutions.

A circuit-generating algebra may be created from a signature by using
:func:`padapto.algebras.trace`. The resulting algebra can be passed to any algorithm to
make it output the underlying algebraic circuits.
"""

from dataclasses import dataclass
from sowing.repr import graphviz
from sowing.node import Node


@dataclass(frozen=True)
class OperatorData:
    """Data attached to a node of an algebraic circuit."""

    # Name of the operator associated to the node
    operator: str

    # List of outdomain arguments for the operator
    args: tuple[Any, ...] = ()

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
    graph_style: graphviz.Style = {}
) -> str:
    """
    Represent a circuit in the DOT format.

    See :func:`sowing.repr.graphviz.write` for details.
    """
    return graphviz.write(circuit, node_style=node_style, graph_style=graph_style)
