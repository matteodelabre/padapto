import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from math import inf

from padapto.algebras import Signature, group, join, lex, limit, pareto, power
from padapto.collections import Multiset, Record


@dataclass(frozen=True)
class EditionSignature[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[T, T], T]
    compare: Callable[[str | None, str | None], T]


def edition[T](
    alg: EditionSignature[T],
    word1: Sequence[str],
    word2: Sequence[str],
) -> T:
    table: dict[tuple[int, int], T] = defaultdict(lambda: alg.null())
    n = len(word1)
    m = len(word2)

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                table[(0, 0)] = alg.unit()
                continue

            change = delete = insert = alg.null()

            if i >= 1 and j >= 1:
                change = alg.combine(
                    table[(i - 1, j - 1)], alg.compare(word1[i - 1], word2[j - 1])
                )

            if i >= 1:
                delete = alg.combine(table[(i - 1, j)], alg.compare(word1[i - 1], None))

            if j >= 1:
                insert = alg.combine(table[(i, j - 1)], alg.compare(None, word2[j - 1]))

            table[(i, j)] = alg.multichoose(change, delete, insert)

    return table[(n, m)]


if __name__ == "__main__":
    # Count the number of possible alignments of two sequences
    # See: Laquer H. Turner, (1981), Asymptotic Limits for a Two-Dimensional Recursion
    # See: OEIS entry A001850
    count = EditionSignature[int](
        null=lambda: 0,
        choose=operator.add,
        unit=lambda: 1,
        combine=operator.mul,
        compare=lambda source, target: 1,
    )

    assert edition(count, "", "") == 1
    assert edition(count, "a", "") == 1
    assert edition(count, "", "b") == 1
    assert edition(count, "a", "b") == 3
    assert edition(count, "ab", "b") == 5
    assert edition(count, "ab", "bc") == 13
    assert edition(count, "abba", "abab") == 321
    assert edition(count, "abcdef", "abcdef") == 8989

    # Generate one of the possible alignments
    type Align = tuple[tuple[str | None, str | None], ...]
    one_align = EditionSignature[Align | None](
        null=lambda: None,
        choose=lambda x, y: y if x is None else x,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
        compare=lambda source, target: ((source, target),),
    )

    assert edition(one_align, "", "") == ()
    assert edition(one_align, "ab", "b") == (("a", None), ("b", "b"))
    assert edition(one_align, "ab", "bc") == (("a", "b"), ("b", "c"))

    # Generate all possible alignments
    all_aligns = one_align | power()

    assert edition(all_aligns, "", "") == Multiset(((),))
    assert edition(all_aligns, "a", "b") == Multiset(
        (
            (("a", "b"),),
            (("a", None), (None, "b")),
            ((None, "b"), ("a", None)),
        )
    )
    assert len(edition(all_aligns, "abcdef", "abcdef")) == 8989
    assert len(set(edition(all_aligns, "abcdef", "abcdef"))) == 8989

    # Compute the minimum cost of an alignment, using unit costs
    def cost_of(align: Align) -> int:
        return sum(1 if source != target else 0 for source, target in align)

    assert cost_of((("a", "a"), ("b", "b"))) == 0
    assert cost_of((("a", "a"), ("b", "c"))) == 1
    assert cost_of((("a", None), ("b", "c"))) == 2
    assert cost_of((("a", None), (None, "a"), ("b", "b"))) == 2

    min_cost = EditionSignature[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
        compare=lambda source, target: 1 if source != target else 0,
    )

    assert edition(min_cost, "ab", "bc") == 2
    assert edition(min_cost, "elephant", "relevant") == 3
    assert edition(min_cost, "abba", "abab") == min(
        cost_of(align) for align in edition(all_aligns, "abba", "abab")
    )

    # Compute the cost of all alignments, in increasing order
    all_min_costs = min_cost | power(order=True)

    assert edition(all_min_costs, "ab", "bc") == Multiset(
        (2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4)
    )
    assert edition(all_min_costs, "abba", "abab") == Multiset(
        cost_of(align) for align in edition(all_aligns, "abba", "abab")
    )

    # Compute the cost of the 5 best alignments
    five_best_costs = min_cost | power(order=True) | limit(5)
    assert edition(five_best_costs, "ab", "bc") == Multiset((2, 2, 3, 3, 3))

    # Compute the number of alignments of minimum cost
    min_cost_count = join(cost=min_cost, count=count) | lex("cost")

    assert edition(min_cost_count, "ab", "bc") == Record(cost=2, count=2)
    assert edition(min_cost_count, "elephant", "relevant") == Record(cost=3, count=2)
    assert edition(min_cost_count, "abba", "abab") == Record(cost=2, count=4)

    # Compute the set of alignments of minimum cost
    min_cost_aligns = join(cost=min_cost, solutions=all_aligns) | lex("cost")

    assert edition(min_cost_aligns, "ab", "bc") == Record(
        cost=2,
        solutions=Multiset(
            (
                (("a", "b"), ("b", "c")),
                (("a", None), ("b", "b"), (None, "c")),
            )
        ),
    )
    assert edition(min_cost_aligns, "elephant", "relevant") == Record(
        cost=3,
        solutions=Multiset(
            (
                (
                    (None, "r"),
                    ("e", "e"),
                    ("l", "l"),
                    ("e", "e"),
                    ("p", "v"),
                    ("h", None),
                    ("a", "a"),
                    ("n", "n"),
                    ("t", "t"),
                ),
                (
                    (None, "r"),
                    ("e", "e"),
                    ("l", "l"),
                    ("e", "e"),
                    ("p", None),
                    ("h", "v"),
                    ("a", "a"),
                    ("n", "n"),
                    ("t", "t"),
                ),
            )
        ),
    )
    assert edition(min_cost_aligns, "abba", "abab") == Record(
        cost=2,
        solutions=Multiset(
            (
                (("a", "a"), ("b", "b"), ("b", "a"), ("a", "b")),
                (("a", "a"), ("b", "b"), (None, "a"), ("b", "b"), ("a", None)),
                (("a", "a"), ("b", None), ("b", "b"), ("a", "a"), (None, "b")),
                (("a", "a"), ("b", "b"), ("b", None), ("a", "a"), (None, "b")),
            )
        ),
    )

    # Compute the number of alignments of each cost
    all_costs_count = join(cost=min_cost, count=count) | power() | group("cost")

    assert edition(all_costs_count, "ab", "bc") == Multiset(
        (
            Record(cost=2, count=2),
            Record(cost=3, count=5),
            Record(cost=4, count=6),
        )
    )
    assert edition(all_costs_count, "abba", "abab") == Multiset(
        (
            Record(cost=key, count=value)
            for key, value in Counter(
                cost_of(align) for align in edition(all_aligns, "abba", "abab")
            ).items()
        )
    )

    # Compute the Pareto-optimal number of operations of each type
    def operations_of(align: Align) -> Record:
        return Record(
            changes=sum(
                (
                    1
                    if source is not None and target is not None and source != target
                    else 0
                )
                for source, target in align
            ),
            deletes=sum(1 if target is None else 0 for source, target in align),
            inserts=sum(1 if source is None else 0 for source, target in align),
        )

    def pareto_filter(vecs: Multiset[Record]) -> Multiset[Record]:
        return Multiset(
            vec
            for vec in set(vecs)
            if not any(
                vec != other
                and all(
                    getattr(other, key) <= getattr(vec, key)
                    for key in ("changes", "deletes", "inserts")
                )
                for other in vecs
            )
        )

    assert operations_of((("a", "a"), ("b", "b"))) == Record(
        changes=0, deletes=0, inserts=0
    )
    assert operations_of((("a", "a"), ("b", "c"))) == Record(
        changes=1, deletes=0, inserts=0
    )
    assert operations_of((("a", None), ("b", "c"))) == Record(
        changes=1, deletes=1, inserts=0
    )
    assert operations_of((("a", None), (None, "a"), ("b", "b"))) == Record(
        changes=0, deletes=1, inserts=1
    )

    min_change = replace(
        min_cost,
        compare=lambda source, target: (
            1 if source is not None and target is not None and source != target else 0
        ),
    )
    min_delete = replace(
        min_cost,
        compare=lambda source, target: 1 if target is None else 0,
    )
    min_insert = replace(
        min_cost,
        compare=lambda source, target: 1 if source is None else 0,
    )
    par_operations = (
        join(changes=min_change, deletes=min_delete, inserts=min_insert)
        | power()
        | pareto("changes", "deletes", "inserts")
    )

    assert edition(par_operations, "", "") == Multiset(
        (Record(changes=0, deletes=0, inserts=0),)
    )
    assert edition(par_operations, "a", "b") == Multiset(
        (
            Record(changes=1, deletes=0, inserts=0),
            Record(changes=0, deletes=1, inserts=1),
        )
    )
    assert edition(par_operations, "elephant", "relevant") == pareto_filter(
        Multiset(
            operations_of(align)
            for align in edition(all_aligns, "elephant", "relevant")
        )
    )
