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
    match: Callable[[str, str], T]
    delete: Callable[[str], T]
    insert: Callable[[str], T]


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
                    table[(i - 1, j - 1)], alg.match(word1[i - 1], word2[j - 1])
                )

            if i >= 1:
                delete = alg.combine(table[(i - 1, j)], alg.delete(word1[i - 1]))

            if j >= 1:
                insert = alg.combine(table[(i, j - 1)], alg.insert(word2[j - 1]))

            table[(i, j)] = alg.multichoose(change, delete, insert)

    return table[(n, m)]


if __name__ == "__main__":
    # Compute the minimum cost of an alignment, using unit costs
    min_cost = EditionSignature[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
        match=lambda sym1, sym2: 1 if sym1 != sym2 else 0,
        delete=lambda sym: 1,
        insert=lambda sym: 1,
    )

    assert edition(min_cost, "ab", "bc") == 2
    assert edition(min_cost, "elephant", "relevant") == 3

    # Compute the cost of all alignments, in increasing order
    all_min_costs = min_cost | power(order=True)

    assert edition(all_min_costs, "ab", "bc") == Multiset(
        (2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4)
    )

    # Compute the cost of the 5 best alignments
    five_best_costs = min_cost | power(order=True) | limit(5)
    assert edition(five_best_costs, "ab", "bc") == Multiset((2, 2, 3, 3, 3))

    # Count the number of possible alignments of two sequences
    # See: Laquer H. Turner, (1981), Asymptotic Limits for a Two-Dimensional Recursion
    # See: OEIS entry A001850
    count = EditionSignature[int](
        null=lambda: 0,
        choose=operator.add,
        unit=lambda: 1,
        combine=operator.mul,
        match=lambda sym1, sym2: 1,
        delete=lambda sym: 1,
        insert=lambda sym: 1,
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
        match=lambda sym1, sym2: (("match", sym1, sym2),),
        delete=lambda sym: (("delete", sym),),
        insert=lambda sym: (("insert", sym),),
    )

    assert edition(one_align, "", "") == ()
    assert edition(one_align, "ab", "b") == (("delete", "a"), ("match", "b", "b"))
    assert edition(one_align, "ab", "bc") == (("match", "a", "b"), ("match", "b", "c"))

    # Generate all possible alignments
    all_aligns = one_align | power()

    assert edition(all_aligns, "", "") == Multiset(((),))
    assert edition(all_aligns, "a", "b") == Multiset(
        (
            (("match", "a", "b"),),
            (("delete", "a"), ("insert", "b")),
            (("insert", "b"), ("delete", "a")),
        )
    )
    assert len(edition(all_aligns, "abcdef", "abcdef")) == 8989
    assert len(set(edition(all_aligns, "abcdef", "abcdef"))) == 8989

    def cost_of(align: Align) -> int:
        return sum(
            0 if kind == "match" and rest[0] == rest[1] else 1 for kind, *rest in align
        )

    assert cost_of((("match", "a", "a"), ("match", "b", "b"))) == 0
    assert cost_of((("match", "a", "a"), ("match", "b", "c"))) == 1
    assert cost_of((("delete", "a"), ("match", "b", "c"))) == 2
    assert cost_of((("delete", "a"), ("insert", "a"), ("match", "b", "b"))) == 2

    assert edition(min_cost, "abba", "abab") == min(
        cost_of(align) for align in edition(all_aligns, "abba", "abab")
    )
    assert edition(all_min_costs, "abba", "abab") == Multiset(
        cost_of(align) for align in edition(all_aligns, "abba", "abab")
    )

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
                (("match", "a", "b"), ("match", "b", "c")),
                (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
            )
        ),
    )
    assert edition(min_cost_aligns, "elephant", "relevant") == Record(
        cost=3,
        solutions=Multiset(
            (
                (
                    ("insert", "r"),
                    ("match", "e", "e"),
                    ("match", "l", "l"),
                    ("match", "e", "e"),
                    ("match", "p", "v"),
                    ("delete", "h"),
                    ("match", "a", "a"),
                    ("match", "n", "n"),
                    ("match", "t", "t"),
                ),
                (
                    ("insert", "r"),
                    ("match", "e", "e"),
                    ("match", "l", "l"),
                    ("match", "e", "e"),
                    ("delete", "p"),
                    ("match", "h", "v"),
                    ("match", "a", "a"),
                    ("match", "n", "n"),
                    ("match", "t", "t"),
                ),
            )
        ),
    )
    assert edition(min_cost_aligns, "abba", "abab") == Record(
        cost=2,
        solutions=Multiset(
            (
                (
                    ("match", "a", "a"),
                    ("match", "b", "b"),
                    ("match", "b", "a"),
                    ("match", "a", "b"),
                ),
                (
                    ("match", "a", "a"),
                    ("match", "b", "b"),
                    ("insert", "a"),
                    ("match", "b", "b"),
                    ("delete", "a"),
                ),
                (
                    ("match", "a", "a"),
                    ("delete", "b"),
                    ("match", "b", "b"),
                    ("match", "a", "a"),
                    ("insert", "b"),
                ),
                (
                    ("match", "a", "a"),
                    ("match", "b", "b"),
                    ("delete", "b"),
                    ("match", "a", "a"),
                    ("insert", "b"),
                ),
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
                1 if kind == "match" and rest[0] != rest[1] else 0
                for kind, *rest in align
            ),
            deletes=sum(1 if kind == "delete" else 0 for kind, *_ in align),
            inserts=sum(1 if kind == "insert" else 0 for kind, *_ in align),
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

    assert operations_of((("match", "a", "a"), ("match", "b", "b"))) == Record(
        changes=0, deletes=0, inserts=0
    )
    assert operations_of((("match", "a", "a"), ("match", "b", "c"))) == Record(
        changes=1, deletes=0, inserts=0
    )
    assert operations_of((("delete", "a"), ("match", "b", "c"))) == Record(
        changes=1, deletes=1, inserts=0
    )
    assert operations_of(
        (("delete", "a"), ("insert", "a"), ("match", "b", "b"))
    ) == Record(changes=0, deletes=1, inserts=1)

    min_change = replace(
        min_cost,
        match=lambda sym1, sym2: 1 if sym1 != sym2 else 0,
        delete=lambda sym: 0,
        insert=lambda sym: 0,
    )
    min_delete = replace(
        min_cost,
        match=lambda sym1, sym2: 0,
        delete=lambda sym: 1,
        insert=lambda sym: 0,
    )
    min_insert = replace(
        min_cost,
        match=lambda sym1, sym2: 0,
        delete=lambda sym: 0,
        insert=lambda sym: 1,
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
