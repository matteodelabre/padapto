# padapto

*padapto* (_Python Algebraic Dynamic Programming Toolkit_) is a Python library designed to facilitate the exploration and design of dynamic programming algorithms in an algebraic style.

## Examples

- [Sequence alignment](examples/align_seq.py)
- [Forest alignment](examples/align_forest.py)

## Installation

_padapto_ is [available as a package on PyPI](https://pypi.org/project/padapto/).
Python ⩾3.12 is required.

```shell
pip install padapto
```

## Usage

To define a new combinatorial problem, one starts by defining its [signature](#signatures).
The signature contains _constructors_ through which objects of the problem's solution space are built.

Instances of these signatures are [evaluation algebras](#evaluation-algebras), which define actions on elements of the search space, such as scoring its elements and looking for optimal solutions.

The recurrences describing how to construct a solution space from a given problem instance can be implemented either through [structure grammars](#structure-grammars), or in a “traditional” way using dynamic programming tables and suitably constructed loops.

Given a solution space, one can use the [`padapto.circuit` module](#solution-spaces) to enumerate and sample solutions according to various distributions.

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

- `multichoose(*args)`: Apply the binary `choose` function to any number of arguments, using the associativity property. For example, `alg.multichoose(a, b, c)` becomes `alg.choose(alg.choose(a, b), c)`.

- `natural_order()`: Create a comparator function ordering elements such that `a <= b` if and only if `alg.choose(a, b) == a`. This is a total and monotone order if the choice function is conservative, i.e., if `alg.choose(a, b)` yields either `a` or `b`.

Subclasses should be frozen dataclasses.
You can add any number of constructors as dataclass fields, whose names must differ from the default functions (`choose`, `null`, `multichoose`, `natural_order`).

### Evaluation algebras

An evaluation algebra is an instance of a signature class for a given _carrier type_ `T`.
Such instances must respect the following five properties:

1. `null` must be a neutral element for `choose` (i.e., `alg.choose(x, alg.null()) == alg.choose(alg.null(), x)) == x` for all values `x` of type `T`).
1. `choose` must be commutative (i.e., `alg.choose(x, y) == alg.choose(y, x)` for all values `x` and `y` of type `T`).
1. `choose` must be associative (i.e., `alg.choose(x, alg.choose(y, z)) == alg.choose(alg.choose(x, y), z)` for all values `x`, `y`, and `z` of type `T`).
1. `choose` must distribute over any constructor (i.e., `alg.f(x, alg.choose(y, z)) == alg.choose(alg.f(x, y), alg.f(x, z))` for any constructor `f` and values `x`, `y`, and `z` of type `T`).
1. `null` must annihilate any constructor (i.e., `alg.f(x, alg.null()) == alg.null()` for any constructor `f` and value `x` of type `T`).

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
> This algebra satisfies all five properties above.
> 
> Using this algebra as-is will just produce a number corresponding to the maximum possible score. The rest of this section describes how to construct actual examples of structures meeting this score.

The `padapto.evaluation` module contains helpers to automatically create valid evaluation algebras and combine them, most of the time saving you the trouble of manually defining them.

The following helpers create algebras:

- `additive(signature, choose, **operators)`:
  Create a cost algebra for the given `signature` where the cost of a constructed object is the sum of the costs of its parts plus an additional term.
  `choose` may be either of the strings `"min"` or `"max"` and determines whether to minimize or maximize the cost (default: `"min"`).
  Each argument in `operators` is a function accepting the arguments that are not sub-solutions and returning the additional constant, for each signature constructor.

- `boltzmann(signature, temperature, **operators)`:
  Create a Boltzmann algebra for the given `signature`, computing the Boltzmann weight of a given set of solutions based on the given additive cost.
  The weights are computed in log-space to avoid underflows.
  `temperature` should be a positive number for minimization and a negative number for maximization.
  The temperature controls how sub-optimal solutions are weighted; when it tends to infinity, all solutions are given the same weight; when it tends to zero, non-optimal solutions get a zero weight.

- `count(signature)`:
  Create a counting algebra for the given `signature`, counting the number of solutions in a given solution space.

- `trace(signature)`:
  Create a tracing algebra for the given `signature`, constructing circuits that represent the traversed [solution space](#solution-spaces).

The following helpers combine algebras:

- `join(**subalgebras)`:
  Combine a set of subalgebras over the same signature into a single joined algebra.
  By default, the carrier type of the resulting algebra is `Record`, and the produced values contain values aggregated from each of the subalgebras, mirroring the keys that were used in the `subalgebras` keyword arguments.
  This carrier type can be changed by passing the `record_type` argument.
  The constructor of the given type will be called by passing the values of the subalgebras as keyword arguments.

- `alg | lex(*keys)`:
  Modify a combined algebra so that the values are compared lexicographically on the given list of fields.
  Each element of `keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields).
  Each of those fields must correspond to an algebra with total and monotone natural orders.
  When two records with the same values on the given keys are compared, their values on the remaining fields are combined using the respective choice functions.

- `alg | power(order=False, unique=False)`:
  Modify an algebra so that its carrier type becomes multisets over the original type.
  When choosing between two multisets of values, the sum of both multisets is taken.
  When constructing values, the original constructors are called on the Cartesian product of all given multisets.
  If `order` is `True`, values are ordered according to the natural order of the original algebra.
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
  Each element of `keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields, and a `*` wildcard can be used to select all fields at a given nesting level).

- `alg | pareto(*keys)`:
  Select non-dominated (Pareto) values produced by the powerset of a joined algebra.
  When choosing between two sets of values, take the union of both and only retain records that are not dominated by than any other record on all of the provided fields (i.e., there does not exist a record that differs on at least one field and that is better than or equal on all fields).
  When two records are equal on the given fields, combine the other fields using the original choice functions.
  Each element of `keys` indicates a field of the combined algebra (in case of nested algebras, dotted notation can be used to access inner fields, and a `*` wildcard can be used to select all fields at a given nesting level).
  Each of those fields must correspond to an algebra with total and monotone natural orders.

### Solution spaces

The `padapto.circuit` module provides utilities for handling algebraic circuits, which are directed acyclic graphs succinctly representing solution spaces.
The `trace` algebra builder from the `padapto.evaluation` module automatically builds such circuits.

- `serialize(circuit)`:
  Transform a circuit to a plain object representation suitable for JSON serialization using the built-in `json` module.

- `unserialize(data)`:
  Reverse the serialization performed by `serialize`.

- `render(circuit, node_style, graph_style, node_metadata)`:
  Produce a GraphViz representation of a circuit in DOT format.
  The `node_style` function maps each node data to a dictionary of GraphViz attributes (default: `default_circuit_style`).
  The `graph_style` dictionary provides GraphViz attributes for the whole graph (default: `None`).
  The `node_metadata` function maps each node identifier to additional metadata to be displayed alongside the node.

- `enumerate_solutions(root)`:
  Generator that yields the solutions encoded by a given circuit one after the other.
  The time and memory required to produce one solution is guaranteed to be linear in the circuit size, however in general there may be exponentially many solutions.

- `get_solution(circuit)`:
  Produce an arbitrary solution from the solutions encoded by the circuit.

- `eval_inside(circuit, alg)`:
  Map each node of the circuit to the value of the subcircuit rooted at that node under a given algebra.
  The keys of the returned mapping are the `id`s of the circuit nodes.

- `eval_outside(circuit, alg, inside, log_weights)`:
  Map each node of the circuit to its _outside_ weight under a given weighting algebra.
  The outside weight of a node is the total weight of all solutions containing that node when treating it as if it were a leaf.
  `inside` should be the result of `eval_inside(circuit, alg)`.
  `log_weights` should be `True` if the weights are represented in log-space.

- `eval(circuit, alg)`:
  Get the value of a circuit under a given algebra.

- `sample(root, gen, weights, log_weights)`:
  Randomly sample a solution from a circuit according to a specified weighting algebra.
  `gen` should be an instance of `random.Random` used for random generation.
  `weights` should either be a weighting algebra, or the result of `eval_inside` on such an algebra.
  `log_weights` should be `True` if the weights are represented in log-space.

### Structure grammars

Structure grammars are a formal system to describe the solution space of a combinatorial optimization problem.
These descriptions are directly executable and can be paired with any valid evaluation algebra to solve such problems.

#### Patterns

Structure grammars rely on a simple pattern-matching engine.
Patterns may be constructed using the following classes of the `padapto.structure` module.

- `Var(name, value)`:
  Match any value and bind it to the variable `name`.
  If `value` is given, match only the given constant.

- **Sequences and multisets**
  - `Empty()`:
    Match the empty sequence.

  - `Item(value, rest)`:
    Decompose a non-empty sequence into its first element, match it against the pattern in `value`, and match the sequence of remaining elements against the pattern in `rest`.

  - `Subseq(value, size, rest)`:
    Decompose a sequence into a prefix and a suffix, match the prefix against the pattern in `value` and the suffix against the pattern in `rest`.
    If `size` is given, only match prefixes of the given sizes.
    The value of `size` may either be a constant number or a range of values described using a `Range(start, stop, step)` object.

  - `Subset(value, size, rest)`:
    Decompose a sequence into two multisets, match one multiset against the pattern in `value` and the other against the pattern in `rest`.
    If `size` is given, only match when the first multiset has the given size.
    The value of `size` may either be a constant number or a range of values described using a `Range(start, stop, step)` object.

- **Natural numbers**
  - `Zero()`:
    Match the number zero.

  - `Term(value, span, rest)`:
    Decompose a natural number into a sum of two terms, match the first term against the pattern in `value` and the second term against the pattern in `rest`.
    If `span` is provided, only match when the first term lies in the given range.
    The value of `span` may either be a constant number or a range of values described using a `Range(start, stop, step)` object.

- **Trees**
  - `Tree(node, edge, parent, children, siblings)`:
    Match a [`sowing`](https://github.com/UdeM-LBIT/sowing) tree cursor.
    The data attached to the pointed node is matched against the `node` pattern and the data attached to its incoming edge is matched against the `edge` pattern.
    If `parent` is provided, only match if the pointed node has a parent, and match this parent against the pattern in `parent`.
    The children of the pointed node are matched against the pattern in `children`.
    Its siblings are matched against the pattern in `siblings`.

The `chain(*patterns)` helper applies to “chainable” patterns, i.e., those with a `rest` attribute, and automatically chains the first pattern to the second, the second to the third, etc.
For example, the following two patterns are equivalent:

```py
pat1 = chain(Subseq(Var("L")), Item(Var("c")), Subseq(Var("R")))
pat2 = Subseq(Var("L"), rest=Item(Var("c"), rest=Subseq(Var("R"))))
```

Once a pattern has been constructed, its `match` method returns a generator over all possible matches.
For example:

```py
>>> print(*pat1.match("abcde"), sep="\n")
{'L': '', 'c': 'a', 'R': 'bcde'}
{'L': 'a', 'c': 'b', 'R': 'cde'}
{'L': 'ab', 'c': 'c', 'R': 'de'}
{'L': 'abc', 'c': 'd', 'R': 'e'}
{'L': 'abcd', 'c': 'e', 'R': ''}
```

#### Grammars, predicates, and clauses

A structure grammar is a rewriting system, over a given problem signature, containing predicates and clauses, that describes how an input corresponds to a set of solutions.
Grammars are declared using classes decorated with `@grammar`.
For example:

```py
@grammar
class FoldGrammar[T](Grammar[T]):
    alg: FoldSignature[T]

    # ... predicates and clauses ...
```
> Declares a grammar for the folding problem. This grammar can be instantiated over any evaluation algebra satisfying the `FoldSignature` problem signature.

Each predicate takes a set of arguments that correspond to parts of the input.
Predicate applications are rewritten using clauses of the grammar.
For example:

```py
@grammar
class FoldGrammar[T](Grammar[T]):
    alg: FoldSignature[T]

    @predicate
    def fold(seq: str) -> T:
        ...

    # ... clauses ...
```
> Declares a predicate called `fold` that accepts a single argument `seq`. Note that the implementation of that predicate will be automatically generated, hence it is not necessary to provide a body for the function.

Each clause describes how a predicate application may be rewritten into a term containing calls to constructors from the signature and new predicate applications.
Note that a predicate can be rewritten by multiple clauses.
For example:

```py
@grammar
class FoldGrammar[T](Grammar[T]):
    # ... signature and predicates ...

    @clause(seq=chain(Subseq(Var("tail")), Item(Var("head"))))
    def _skip(self, tail: str, head: str) -> T:
        return self.alg.skip(self.fold(seq=tail), head)

    # ... other clauses ...
```
> Declares a clause that rewrites a call to `fold(x)` to `skip(fold(y), c)`, where `x` has been decomposed into its first letter `c` and the rest `y`. It is not necessary in this case to specify that the clause applies to the `fold` predicate as this grammar has only one predicate.

Once a grammar like `FoldGrammar` has been declared, it can be instantiated over any algebra that matches its signature.
The predicates of the grammar can then be called directly, providing the desired input as arguments.

```py
>>> gram = FoldGrammar(alg=max_score)
>>> gram.fold(seq="ACACUUGGC")
3
```

Conceptually, the obtained result corresponds to performing all possible rewritings of the predicate to terms of the signature, evaluating each of those terms under the given algebra, and combining them using the algebra's choice function.
Using the `max_score` algebra, this corresponds to generating all possible secondary structures of the given string, counting the number of base pairs in each, and returning the maximum number of pairs.

Internally, the generated code takes advantage of the recursive description and the distributivity property of the evaluation algebra (equivalent to Bellman's principle) to turn this exponential-time computation into a cubic-time algorithm.
In effect, we have automatically generated [Nussinov](https://en.wikipedia.org/wiki/Nussinov_algorithm)'s algorithm from a high-level description!

The `@grammar`, `@predicate` and `@clause` decorators act jointly to generate code.
The constructed memoization tables can be inspected using the internal `memo` attribute of the grammar instance.

## References

The main influences of this work are:

- R. Giegerich, C. Meyer, and P. Steffen, [“A discipline of dynamic programming over sequence data,”](https://doi.org/10.1016/j.scico.2003.12.005) Science of Computer Programming, vol. 51, no. 3, pp. 215–263, Jun. 2004.
- P. Steffen and R. Giegerich, [“Versatile and declarative dynamic programming using pair algebras,”](https://doi.org/10.1186/1471-2105-6-224) BMC Bioinformatics, vol. 6, no. 1, Dec. 2005.
- C. Saule and R. Giegerich, [“Pareto optimization in algebraic dynamic programming,”](https://almob.biomedcentral.com/articles/10.1186/s13015-015-0051-7) Algorithms Mol Biol, vol. 10, no. 1, Dec. 2015.
- M. Riechert, C. H. zu Siederdissen, and P. F. Stadler, [“Algebraic dynamic programming for multiple context-free grammars,”](https://doi.org/10.1016/j.tcs.2016.05.032) Theoretical Computer Science, vol. 639, pp. 91–109, Aug. 2016.
- S. Berkemer, C. H. zu Siederdissen, and P. Stadler, [“Algebraic dynamic programming on trees,”](https://doi.org/10.3390/a10040135) Algorithms, vol. 10, no. 4, p. 135, Dec. 2017.

Other existing implementations of the _algebraic_ paradigm for dynamic programming include:

- [ADPfusion](https://github.com/choener/ADPfusion), a Haskell library
- [Bellman's GAP](https://github.com/jlab/gapc), a C++ compiler for a DSL

## License

This code is released under the [GNU General Public License v3](./LICENSE), or, at your option, any later version.
