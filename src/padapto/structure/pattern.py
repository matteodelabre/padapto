from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from itertools import combinations, product
from typing import Any, Protocol, Self

from sowing import Hedge, Zipper


class Pattern[T](Protocol):
    """Pattern that can be matched against an input to retrieve variable bindings."""

    def match(self, data: T) -> Iterable[Mapping[str, Any]]:
        """
        Match an input against the current pattern.

        :param data: input to match
        :returns: mapping of each pattern variable to its bound value
        """
        ...


class ChainablePattern[T](Pattern[T], Protocol):
    """Extract part of a sequence and delegate to another pattern for the rest."""

    rest: Pattern[T]


def chain[T](
    head: ChainablePattern[T], *patterns: ChainablePattern[T]
) -> ChainablePattern[T]:
    """Chain multiple sequence patterns one after the other."""
    if not patterns:
        return head

    return replace(head, rest=chain(*patterns))  # type: ignore[type-var]


def merge[Key, Value](
    *iterables: Iterable[Mapping[Key, Value]] | None
) -> Iterable[Mapping[Key, Value]]:
    """
    Produce combined assignments from a set of assignment iterables.

    :param iterables: list of assignment iterable to combine
    :returns: combined assignment for each element of the product of the input
    """
    for comb in product(*(iterable for iterable in iterables if iterable is not None)):
        yield reduce(lambda left, right: {**left, **right}, comb)


@dataclass(frozen=True, slots=True)
class Var[T](Pattern[T]):
    """Match any input completely and optionally store it in a variable."""

    name: str | None = None
    """If set, the matched input is bound to the given variable name."""

    value: T | None = None
    """If set, the input is matched only if it equals the given specific value."""

    def match(self, data: T) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if self.value is not None and self.value != data:
            return

        if self.name is not None:
            yield {self.name: data}
        else:
            yield {}


@dataclass(frozen=True, slots=True)
class Empty[T](Pattern[Sequence[T]]):
    """Match the empty sequence."""

    def match(self, data: Sequence[T]) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if not data:
            yield {}


@dataclass(frozen=True, slots=True)
class Item[T](ChainablePattern[Sequence[T]]):
    """Extract the first item from a non-empty sequence."""

    value: Pattern[T] = Var()
    """Pattern matching the first element of the sequence."""

    rest: Pattern[Sequence[T]] = Empty()
    """Pattern matching the rest of the sequence."""

    def match(self, seq: Sequence[T]) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if not seq:
            return

        yield from merge(self.value.match(seq[0]), self.rest.match(seq[1:]))


@dataclass(frozen=True, slots=True)
class Range:
    """Range of non-negative integer values with optional upper bound."""

    start: int = 0
    """Minimum value enumerated by this range."""

    stop: int | None = None
    """
    Value at which this range stops.
    If None, stops at any bound provided in :meth:`bound`.
    """

    step: int | None = None
    """Increments by which to traverse the range."""

    @classmethod
    def of(cls, init: int | Self) -> Self:
        """Optionally convert a single integer to a range containing it."""
        if isinstance(init, int):
            return cls(init, init + 1)
        else:
            return init

    def bound(self, upper: int) -> Iterable[int]:
        """Return an iterable spanning this range up to a given bound."""
        lower = max(0, self.start)

        if self.stop is not None:
            upper = min(self.stop, upper)

        if self.step is not None:
            return range(lower, upper, self.step)

        return range(lower, upper)


@dataclass(frozen=True, slots=True)
class Subseq[T](ChainablePattern[Sequence[T]]):
    """Extract any prefix from a sequence."""

    value: Pattern[Sequence[T]] = Var()
    """Pattern matching the sequence prefix."""

    size: int | Range = Range()
    """Only match prefixes of the given lengths (default: any length is allowed)."""

    rest: Pattern[Sequence[T]] = Empty()
    """Pattern matching the rest of the sequence."""

    def match(self, seq: Sequence[T]) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if self.size == Range() and isinstance(self.value, Item):
            # Only try single-item lists if the expected value is a single item
            size = Range.of(1)
        else:
            size = Range.of(self.size)

        values = size.bound(len(seq) + 1)

        if self.rest == Empty():
            if len(seq) in values:
                yield from self.value.match(seq)
        else:
            for k in values:
                yield from merge(self.value.match(seq[:k]), self.rest.match(seq[k:]))


def split_set_size[Elt](
    elts: Sequence[Elt],
    size: int,
) -> Iterable[tuple[Sequence[Elt], Sequence[Elt]]]:
    """
    Enumerate all bipartitions of a multiset with a given left-hand set size.

    :param elts: original multiset
    :param size: size of the left-hand set
    :returns: enumerates all bipartitions in lexicographical order of the input
    """
    left = combinations(elts, r=size)
    right = list(combinations(elts, r=len(elts) - size))
    return zip(left, right[::-1], strict=True)


@dataclass(frozen=True, slots=True)
class Subset[T](ChainablePattern[Sequence[T]]):
    """Extract a submultiset from a sequence, viewed as a multiset."""

    value: Pattern[Sequence[T]] = Var()
    """Pattern matching the submultiset."""

    size: int | Range = Range()
    """Only match multisets of the given size (default: any size is allowed)."""

    rest: Pattern[Sequence[T]] = Empty()
    """Pattern matching the rest of the multiset."""

    def match(self, elts: Sequence[T]) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if self.size == Range() and isinstance(self.value, Item):
            # Only try singletons if the expected value is a single item
            size = Range.of(1)
        else:
            size = Range.of(self.size)

        values = size.bound(len(elts) + 1)

        if self.rest == Empty():
            if len(elts) in values:
                yield from self.value.match(elts)
        else:
            for k in values:
                for left, right in split_set_size(elts, k):
                    yield from merge(self.value.match(left), self.rest.match(right))


@dataclass(frozen=True, slots=True)
class Zero(Pattern[int]):
    """Match the number zero."""

    def match(self, data: int) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        if data == 0:
            yield {}


@dataclass(frozen=True, slots=True)
class Term(ChainablePattern[int]):
    """Extract a non-negative summand from a number."""

    value: Pattern[int] = Var()
    """Pattern matching the summand."""

    span: int | Range = Range()
    """Only values inside these bounds are matched (default: any non-negative value)."""

    rest: Pattern[int] = Zero()
    """Pattern matching the rest of the number (default: rest is zero)."""

    def match(self, data: int) -> Iterable[Mapping[str, Any]]:  # noqa: D102
        values = Range.of(self.span).bound(data + 1)

        if self.rest == Zero():
            if data in values:
                yield from self.value.match(data)
        else:
            for head in values:
                yield from merge(self.value.match(head), self.rest.match(data - head))


@dataclass(frozen=True, slots=True)
class Tree[NodeData, EdgeData](Pattern[Zipper[NodeData, EdgeData]]):
    """Decompose a tree."""

    node: Pattern[NodeData] = Var()
    """Pattern matching the data attached to the tree root."""

    edge: Pattern[EdgeData | None] = Var()
    """Pattern matching the data attached to the incoming edge of the root."""

    parent: Pattern[Zipper[NodeData, EdgeData]] | None = None
    """If set, only matches nodes that have a parent through this pattern."""

    children: Pattern[Sequence[Zipper[NodeData, EdgeData]]] = Var()
    """Pattern matching the sequence of children of the root."""

    siblings: Pattern[Sequence[Zipper[NodeData, EdgeData]]] = Var()
    """Pattern matching the sequence of siblings of the node from left to right."""

    def match(self, cursor: Zipper[NodeData, EdgeData]):  # noqa: D102
        if cursor.node is None:
            return

        parent_it: Iterable[Mapping[str, Any]] = ({},)

        if cursor.is_root():
            if self.parent is not None:
                return
        else:
            if self.parent is not None:
                parent_it = self.parent.match(cursor.up())

        children: Sequence[Zipper[NodeData, EdgeData]] = Hedge()

        if self.children != Var() and not cursor.is_leaf():
            children = Hedge(cursor.down(), len(cursor.node.edges))

        siblings: Sequence[Zipper[NodeData, EdgeData]] = ()

        if self.siblings != Var() and not cursor.is_root():
            prev_siblings = Hedge(cursor.up().down(), cursor.index)
            next_siblings: Hedge[NodeData, EdgeData]

            if cursor.is_last_sibling():
                next_siblings = Hedge()
            else:
                assert cursor.parent is not None
                assert cursor.parent.node is not None
                next_siblings = Hedge(
                    cursor.up().down(cursor.index + 1),
                    len(cursor.parent.node.edges) - cursor.index - 1,
                )

            siblings = tuple(prev_siblings) + tuple(next_siblings)

        yield from merge(
            self.node.match(cursor.node.data),
            self.edge.match(cursor.data),
            parent_it,
            self.children.match(children),
            self.siblings.match(siblings),
        )
