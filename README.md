# padapto

*padapto* (_Python Algebraic Dynamic Programming Toolkit_) is a Python library designed to facilitate the exploration and design of dynamic programming algorithms in an algebraic style.

## Examples

- [Sequence alignment](examples/align_seq.py)
- [Forest alignment](examples/align_forest.py)

## Usage

To define a new combinatorial problem, one starts by defining its [signature](#signatures).
The signature contains _constructors_ through which objects of the problem search space are spawned.

Instances of these signatures are [evaluation algebras](#evaluation-algebras), which define actions on elements of the search space, such as scoring its elements and looking for optimal solutions.

The recurrences describing how to construct a search space from a given problem instance can be implemented either through [structure grammars](#structure-grammars), or in a “traditional” way using dynamic programming tables and suitably-constructed loops.

Given a specific solution search space, one can use the [`padapto.circuit` module](#solution-spaces) to enumerate and sample solutions according to various distributions.

### Signatures

To define a signature, **subclass the `Signature` base class** from the `padapto.signature` module.

```py
@dataclass(frozen=True)
class FoldSignature[T](Signature[T]):
    unit: Callable[[], T]
    pair: Callable[[T, T, str, str], T]
    skip: Callable[[T, str], T]
```

> This example defines a signature for the [RNA folding problem](https://en.wikipedia.org/wiki/Nucleic_acid_structure_prediction#Dynamic_programming_algorithms).
> The `unit` constructor takes no argument and produces the empty structure.
> The `pair` constructor takes two substructures (the arguments of type `T`) and two characters (type `str`) corresponding to the paired nucleotides.
> The `skip` constructor takes one substructure and one nucleotide to be skipped.

The `Signature` base class provides fields for the choice function (`choose`) and the neutral element (`null`), which all signatures must contain.
It also defines the following helper methods:

- `multichoose(*args)`: Apply the binary `choose` function to any number of arguments, using the associativity property. For example `alg.multichoose(a, b, c)` becomes `alg.choose(alg.choose(a, b), c)`.
- `natural_order()`: Create a comparator function ordering elements such that `a <= b` if and only if `alg.choose(a, b) == a`. This is a total and monotonous order if the choice function is conservative, i.e., if `alg.choose(a, b)` yields either `a` or `b`.

Subclasses should be frozen dataclasses.
You can add any number of constructors as dataclass fields, which must have different names from the default functions (`choose`, `null`, `multichoose`, `natural_order`).

### Evaluation algebras

An evaluation algebra is an instance of a signature class for a given _carrier type_ `T`.
Such instances must respect the following five properties:

1. `null` must be a neutral element for `choose` (i.e., `alg.choose(x, alg.null()) == alg.choose(alg.null(), x)) == x` for all values `x` of type `T`).
1. `choose` must be commutative (i.e., `alg.choose(x, y) == alg.choose(y, x)` for all values `x` and `y` of type `T`).
1. `choose` must be associative (i.e., `alg.choose(x, alg.choose(y, z))) == alg.choose(alg.choose(x, y), z)` for all values `x`, `y`, and `z` of type `T`).
1. `choose` must distribute over any constructor (i.e., `alg.f(x, alg.choose(y, z)) == alg.choose(alg.f(x, y), alg.f(x, z)))` for any constructor `f` and values `x`, `y`, and `z` of type `T`.
1. `null` must annihilate any constructor (i.e., `alg.f(x, alg.null()) == alg.null()` for any constructor `f` and value `x` of type `T`.

**There is no automated check for these properties,** but using algebras that violate them may produce unexpected or invalid results.

```py
max_score = FoldSignature[int | float](
    null=lambda: -inf,
    choose=max,
    unit=lambda: 0,
    pair=lambda cost1, cost2, sym1, sym2: cost1 + cost2 + 1,
    skip=lambda cost, sym: cost,
)
```

> This evaluation algebra gives a score to each secondary structure corresponding to the number of paired nucleotides that it contains.
> It also chooses the maximum score among all solutions.
> This algebra satisfies all of the five properties above.
> 
> Using this algebra as-is will just produce a number corresponding to the maximum possible score. To know how to construct an actual example of a structure meeting this score, read on.

The `padapto.evaluation` module contains helpers to automatically create valid evaluation algebras and combine them, most of the times saving you the trouble of manually defining them.

The following helpers create algebras:

- `additive(signature, choose, **operators)`:
  Create a cost algebra for the given `signature` where the cost of a constructed object is the sum of the costs of its parts with some additional constant.
  `choose` may be either of the strings `"min"` or `"max"` and determines whether to minimize or maximize the cost (default: min).
  Each argument in `operators` is a function accepting the arguments that are not sub-solutions and returning the additional constant, for each signature constructor.

- `boltzmann(signature, temperature, **operators)`:
  Create a Boltzmann algebra for the given `signature`, computing the Boltzmann weight of a given set of solutions based on the given additive cost.
  The weights are computed in log-space to avoid underflows.
  `temperature` should be a positive number for minimization and a negative number for maximization.
  The temperature controls how sub-optimal solutions are weighted; when it goes towards infinity, all solutions are given the same weight; when it goes towards zero, non-optimal solutions get a null weight.

- `count(signature)`:
  Create a counting algebra for the given `signature`, counting the number of solutions in a given solution space.

- `trace(signature)`:
  Create a tracing algebra for the given `signature`, constructing circuits that represent the traversed [solution space](#solution-spaces).

The following helpers combine algebras:

- `join(**subalgebras)`:
  Combine a set of subalgebras over the same signature into a single joined algebra.
  By default, the carrier type of the resulting algebra is `Record`, and the produced values contain values agregated from each of the subalgebras, mirroring the keys that were used in the `subalgebras` keyword arguments.
  This carrier type can be changed by passing the `record_type` argument.
  The constructor of the given type will be called by passing the values of the subalgebras as keyword arguments.

- `alg | lex(*keys)`:
  Modify a combined algebra so that the values are compared lexicographically on the given list of fields.
  Each element of `*keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields).
  Each of those fields must correspond to an algebra with total and monotonous natural orders.
  When two records with the same values on the given keys are compared, their values on the remaining fields are combined using the respective choice functions.

- `alg | power(order=False, unique=False)`:
  Modify an algebra so that its carrier type is the multi-powerset of the original type.
  When choosing between two multisets of values, the sum of both multisets is taken.
  When constructing values, the original constructors are called on the Cartesian product of all given multisets.
  If `order` is `True`, values are ordered against the natural order of the original algebra.
  A custom comparator function can also be passed to `order` to use another order.
  If `unique` is `True`, duplicate values are removed from the multisets after each choice or construction operation.

- `alg | limit(maxsize)`:
  Limit the size of multisets produced by a power algebra to the given `maxsize`.
  If `maxsize` is `None`, then this is a no-op (the size stays unlimited).
  Note that this produces **invalid algebras** when `maxsize` is not `0`, `1`, or `None`, as the algebra fails the distributivity criterion.
  This can still be useful to produce partial results for problems with large numbers of results.

- `alg | group(*keys)`:
  Group values produced by the powerset of a joined algebra so that there are no duplicates on the given fields.
  When choosing between two values that are duplicates, the values on the remaining fields are combined using the original choice functions.
  Each element of `*keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields, and a `*` wildcard can be used to select all fields at a given nesting level).

- `alg | pareto(*keys)`:
  Select non-dominated (Pareto) values produced by the powerset of a joined algebra.
  When choosing between two sets of values, take the union of both and only retain records that are not strictly worse than any other record on all of the provided fields.
  When two values are equal on the given fields, combine the other fields using the original choice functions.
  Each element of `*keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields, and a `*` wildcard can be used to select all fields at a given nesting level).
  Each of those fields must correspond to an algebra with total and monotonous natural orders.

### Solution spaces

The `padapto.circuit` module provides utilities for handling algebraic circuits, which are directed acyclic graphs succinctly representing solution spaces.
The `trace` algebra builder from the `padapto.evaluation` module automatically builds such circuits.

- `serialize(circuit)`:
  Transforms a circuit to a plain object representation suitable for JSON serialization using the built-in `json` module.

- `unserialize(data)`:
  Reverse the serialization performed by `serialize`.

- `render(circuit, node_style, graph_style, node_metadata)`:
  Produce a GraphViz representation of a circuit in DOT format.
  The `node_style` function maps each node data to a dictionary of GraphViz attributes (default: `default_circuit_style`).
  The `graph_style` dictionary provides GraphViz attributes for the whole graph (default: `None`).
  The `node_metadata` function maps each node identifier to additional metadata to be displayed alongside the node.

- `enumerate_solutions(root)`:
  Generator that yields the solutions encoded by a given circuit one after the other.
  The time and memory required to produce one solution is guaranteed to be linear in the circuit size, however in general there may be an exponential number of solutions.

- `get_solution(circuit)`:
  Produce an arbitrary solution from the solutions encoded by the circuit.

- `eval_inside(circuit, alg)`:
  Map each node of the circuit to the value of the subcircuit starting at that node under a given algebra.
  The keys of the returned mapping are the `id`s of the circuit nodes.

- `eval_outside(circuit, alg, inside, log_weights)`:
  Map each node of the circuit to its _outside_ weight under a given weighting algebra.
  The outside weight of a node is the total weight of all solutions containing that node when treating it as if it were a leaf.
  `inside` should be the result of `eval_inside(circuit, alg)`.
  `log_weights` should be true if the weights are represented in log-space.

- `eval(circuit, alg)`:
  Get the value of a circuit under a given algebra.

- `sample(root, gen, weights, log_weights)`:
  Randomly sample a solution from a circuit according to a specified weighting algebra.
  `gen` should be an instance of `random.Random` used for random generation.
  `weights` should either be a weighting algebra, or the result of `eval_inside` on such an algebra.
  `log_weights` should be true if the weights are represented in log-space.

### Structure grammars

```py
@grammar
class FoldGrammar[T](Grammar[T]):
    alg: FoldSignature[T]

    @predicate
    @staticmethod
    def fold(seq: str) -> T:
        ...

    @clause(seq=Empty())
    def _empty(self):
        return self.alg.unit()

    @clause(
        seq=chain(
            Subseq(Var("tail")),
            Item(Var("left")),
            Subseq(Var("inner")),
            Item(Var("right")),
        ),
    )
    def _pair(self, tail: str, left: str, inner: str, right: str) -> T:
        if {left, right} in ({"A", "U"}, {"C", "G"}):
            return self.alg.pair(
                self.fold(seq=tail),
                self.fold(seq=inner),
                left,
                right,
            )

        return self.alg.null()

    @clause(seq=chain(Subseq(Var("tail")), Item(Var("head"))))
    def _skip(self, tail: str, head: str) -> T:
        return self.alg.skip(self.fold(seq=tail), head)
```

TODO

## License

This code is released under the [GNU General Public License v3](./LICENSE) license, or any newer version of the GPL license.
