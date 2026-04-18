"""
Rewrite systems describing solution space structures.

Grammars provide a generic way to solve combinatorial problems by defining rewrite
rules ("clauses") that process inputs and turn them into algebraic terms representing
the output. Such systems are automatically turned into effective and efficient
algorithms by the procedures of this module.

To define a grammar, create a class decorated with @:func:`grammar` and containing a
field named `alg`. Clauses of the grammar are defined as (usually private) methods
decorated with @:func:`clause`. Predicates of the grammar are (usually public) fields
declared using :func:`predicate`.

To instanciate a grammar, simply call its constructor and provide a suitable evaluation
algebra. Interactions with the resulting grammar usually happen through its predicates.
Each grammar instance has its own independent memoization tables.
"""

import inspect
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableMapping, Sequence, Set
from dataclasses import dataclass
from functools import partialmethod
from inspect import Parameter
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

from immutables import Map

from ..algebras import Signature
from .pattern import Pattern, merge


@runtime_checkable
class Clause[Out](Protocol):
    """Clause of a grammar."""

    __name__: str
    """Name of the clause."""

    predicates: Set[str]
    """
    Name of the predicates matched by the clause (an empty set matches any predicate).
    All the matched predicates must accept the same set of inputs.
    """

    head: Mapping[str, Pattern[Any]]
    """
    Set of patterns matching each input of the predicate by name. When parsing, the
    pattern variables resulting from matching these patterns will be available as
    arguments to the clause.
    """

    def __call__(gram, *args: Any) -> Out:
        """Invoke the clause with the given instanciated variables."""
        ...


def clause[Out](
    predicate: str | None = None,
    predicates: Set[str] = frozenset(),
    **patterns: Pattern[Any],
) -> Callable[[Callable[..., Out]], Clause[Out]]:
    """
    Decorate a grammar clause method.

    See :class:`Clause` for details.
    """

    def decorator(body: Callable[..., Out]) -> Clause[Out]:
        body = cast(Clause[Out], body)
        body.predicates = predicates

        if predicate is not None:
            body.predicates |= {predicate}

        body.head = patterns
        return body

    return decorator


@dataclass(frozen=True, slots=True)
class Predicate:
    """Predicate of a grammar."""

    name: str
    """Name of the predicate."""

    doc: str | None
    """Docstring of the predicate."""

    parameters: Mapping[str, Parameter]
    """Set of parameters of the predicate."""


def predicate[T: Callable](fun: T) -> T:
    """
    Decorate a grammar predicate method.

    The original function body is discarded and will be replaced by an actual
    implementation by the @:func:`grammar` decorator.

    See :class:`Predicate` for details.
    """
    return cast(
        T,
        Predicate(
            name=fun.__name__,
            doc=fun.__doc__,
            parameters=inspect.signature(fun).parameters,
        ),
    )


class Grammar[Out](Protocol):
    """
    Yield grammar encoded as a class.

    To create grammar classes, use the @:func:`grammar` decorator.
    """

    def __init__(self, alg: Signature[Out]):
        """Create a grammar instance on the given algebra."""
        return

    alg: Signature[Out]
    """Algebra for the produced solution terms."""

    memo: Mapping[str, MutableMapping[Mapping, Out]] = {}
    """Memoization tables indexed by predicate and by input."""

    predicates: ClassVar[Mapping[str, Predicate]] = {}
    """Set of grammar predicates associated with their signature."""

    clauses: ClassVar[Mapping[str, Sequence[Clause[Out]]]] = {}
    """Set of grammar clauses indexed by their predicate."""


def _gram_parse[Out](self: Grammar[Out], pred: str, **args: Any) -> Out:
    """
    Perform parsing of an input under a predicate.

    :param self: grammar holding the rewrite rules
    :param pred: predicate to parse under
    :param args: input data
    :returns: parsing result as an algebra value
    """
    args_key = Map(args)

    if args_key in self.memo[pred]:
        return self.memo[pred][args_key]

    result = self.alg.null()

    for clause in self.clauses[pred]:
        for assign in merge(
            *(pattern.match(args[name]) for name, pattern in clause.head.items())
        ):
            result = self.alg.choose(result, clause(self, **assign))

    self.memo[pred][args_key] = result
    return result


def _gram_init[Out](self, alg: Signature[Out]):
    """Initialize a grammar instance."""
    object.__setattr__(self, "alg", alg)
    object.__setattr__(self, "memo", defaultdict(dict))


def grammar[Out](cls) -> Grammar[Out]:
    """Decorate a grammar class."""
    # Collect predicates and inject their implementation
    all_preds: dict[str, Predicate] = {}
    cls.predicates = all_preds

    for pred in cls.__dict__.values():
        if isinstance(pred, Predicate):
            all_preds[pred.name] = pred
            setattr(cls, pred.name, partialmethod(_gram_parse, pred.name))

    # Collect clauses and check compatibility
    clauses: dict[str, list[Clause[Out]]] = defaultdict(list)
    cls.clauses = clauses

    for clause in cls.__dict__.values():
        if isinstance(clause, Clause):
            clause_preds = clause.predicates if clause.predicates else all_preds

            for pred_name in clause_preds:
                pred = all_preds[pred_name]
                extra = clause.head.keys() - pred.parameters.keys()
                missing = pred.parameters.keys() - clause.head.keys()

                if extra:
                    raise TypeError(
                        f"clause '{clause.__name__}' expects input(s) not provided by "
                        f"predicate '{pred_name}': {', '.join(extra)}"
                    )

                if missing:
                    raise TypeError(
                        f"clause '{clause.__name__}' ignores input(s) provided by "
                        f"predicate '{pred_name}': {', '.join(missing)}"
                    )

                clauses[pred_name].append(clause)

    # Create dataclass and inject grammar initializer
    cls.__init__ = _gram_init
    return cls
