"""
An algebraic circuit is an encoding of a solution space as a DAG.

In most cases, this encoding allows representing an exponential number of solutions
inside a polynomial-sized object and to efficiently extract solutions and compute
summary information on the set of solutions.

See also :func:`padapto.algebras.trace` to create circuit-generating algebras from a
given signature.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from random import Random
from typing import TYPE_CHECKING, Any, TypeVar, get_args

from immutables import Map
from sowing import Edge, Node, traversal
from sowing.repr import graphviz

if TYPE_CHECKING:
    from .algebras.signature import Signature


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


def _passthrough[T](value: T) -> T:
    return value


def serialize(
    circuit: Circuit, arg_encoder: Callable = _passthrough
) -> Mapping[str, Any]:
    """
    Encode an algebraic circuit as a flat and plain JSON structure.

    :param circuit: circuit to be encoded
    :param arg_encoder: function transforming each argument into a plain JSON value
        (default: identity function)
    :returns: encoded object, to use in :func:`json.dump` or :func:`json.dumps`
    """
    ids: dict[int, str] = {}
    result: dict[str, dict[str, Any]] = {}

    def seq_id(node: Node) -> str:
        if id(node) not in ids:
            ids[id(node)] = str(len(ids))

        return ids[id(node)]

    for cursor in traversal.depth(circuit, preorder=True, unique="id"):
        node = cursor.node
        assert node is not None
        data = node.data

        attrs: dict[str, Any] = {"operator": data.operator}
        result[seq_id(node)] = attrs

        if data.args:
            attrs["args"] = [arg_encoder(arg) for arg in data.args]

        if node.edges:
            attrs["children"] = [seq_id(edge.node) for edge in node.edges]

    return result


def _circuit_toposort(data: Mapping[str, Any]) -> Iterable[str]:
    parents = defaultdict(set)
    outdeg = {}
    candidates = []

    for seq_id, attrs in data.items():
        children = attrs.get("children", ())
        outdeg[seq_id] = len(children)

        if len(children) == 0:
            candidates.append(seq_id)

        for child in children:
            parents[child].add(seq_id)

    while candidates:
        current = candidates.pop()
        yield current

        for parent in parents[current]:
            outdeg[parent] -= 1

            if outdeg[parent] == 0:
                candidates.append(parent)

    if sum(outdeg.values()) > 0:
        raise ValueError("circuit contains a cycle")


def unserialize(
    data: Mapping[str, Any], arg_decoder: Callable = _passthrough
) -> Circuit:
    """
    Decode an algebraic circuit from its JSON representation.

    :param circuit: decoded JSON object from :func:`json.load` or :func:`json.loads`
    :param arg_decoder: function transforming arguments from their plain JSON value
        representation back to their original form
        (default: identity function)
    :returns: decoded circuit
    """
    if "0" not in data:
        raise ValueError("circuit does not contain a root node #0")

    result: dict[str, Circuit] = {}

    for item in _circuit_toposort(data):
        attrs = data[item]
        result[item] = Node(
            data=OperatorData(
                operator=attrs["operator"],
                args=tuple(arg_decoder(arg) for arg in attrs.get("args", ())),
            ),
            edges=tuple(Edge(result[child]) for child in attrs.get("children", ())),
        )

    return result["0"]


def default_circuit_style(data: OperatorData, meta: Any) -> graphviz.Style:
    """Render circuit nodes as GraphViz nodes."""
    attrs = {}

    if data.is_choose():
        attrs["label"] = "⊕"
        attrs["shape"] = "none"
        attrs["width"] = "0"
        attrs["height"] = "0"

        if meta is not None:
            attrs["label"] += "\n" + str(meta)
    else:
        head = data.operator
        args = data.args
        op_label = f"{head}({', '.join(map(repr, args))})" if args else head

        # Preserve left-to-right order of outgoing edges
        attrs["ordering"] = "out"

        if meta is None:
            attrs["shape"] = "box"
            attrs["style"] = "rounded"
            attrs["label"] = op_label
        else:
            attrs["shape"] = "Mrecord"
            attrs["label"] = f"{{ {op_label} | {meta} }}"

    return attrs


def render(
    circuit: Circuit,
    node_style: Callable[[OperatorData, Any], graphviz.Style] = default_circuit_style,
    graph_style: graphviz.Style | None = None,
    node_metadata: Mapping[int, Any] = Map(),
) -> str:
    """
    Represent a circuit in the DOT format.

    See :func:`sowing.repr.graphviz.write` for details.

    :param node_style: mapping from each operator data to a dictionary of GraphViz
        attributes
    :param graph_style: dictionary of GraphViz attributes for the whole graph
    :param node_metadata: dictionary of additional metadata to display alongside each
        node, indexed by the node identifier
    """
    if graph_style is None:
        graph_style = {}

    def wrapped_node_style(node: Circuit) -> graphviz.Style:
        return node_style(node.data, node_metadata.get(id(node), None))

    return graphviz.write(
        circuit, node_style=wrapped_node_style, graph_style=graph_style
    )


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

    inside: dict[int, T] = {}

    for cursor in traversal.depth(circuit, preorder=False, unique="id"):
        node = cursor.node
        assert node is not None

        op_data = node.data
        children_values = [inside[id(child)] for child in node.children()]

        if op_data.is_choose():
            value = alg.multichoose(*children_values)
        else:
            # Collect and reorder children arguments and outdomain arguments
            signature = signatures[op_data.operator]
            args = []
            next_child = 0
            next_outdomain = 0

            for kind in signature:
                if isinstance(kind, TypeVar):
                    args.append(children_values[next_child])
                    next_child += 1
                else:
                    args.append(op_data.args[next_outdomain])
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


def eval_outside(
    circuit: Circuit, alg: Signature[float], inside: dict[int, float]
) -> dict[int, float]:
    """
    Evaluate the outside weight of each node in a circuit.

    For each circuit node, its outside weight is the total weight of all solutions
    containing the node when treating this node as if it were a leaf.

    :param circuit: circuit describing the circuit to evaluate
    :param alg: weighting algebra used for evaluaiton
    :param inside: inside values as computed by :func:`eval_inside`
    :returns: dictionary associating each node ID to its outside value
    """
    outside: dict[int, float] = defaultdict(float)
    outside[id(circuit)] = 1

    for cursor in traversal.topological(circuit):
        node = cursor.node
        assert node is not None

        value = outside[id(node)]

        if node.data.is_choose():
            for child in cursor.children():
                outside[id(child.node)] += value
        else:
            for child in cursor.children():
                outside[id(child.node)] += (
                    value * inside[id(node)] / inside[id(child.node)]
                )

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
    root: Circuit, gen: Random, weights: Signature[float] | Mapping[int, float]
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
        return Node[OperatorData, None](root.data).extend(
            sample(child, gen, weights) for child in children
        )
