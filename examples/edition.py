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

    assert edition(min_cost, "", "") == 0
    assert edition(min_cost, "ab", "bc") == 2
    assert edition(min_cost, "abba", "abab") == 2
    assert edition(min_cost, "alberta", "camera") == 4

    # Compute the cost of all alignments, in increasing order
    all_min_costs = min_cost | power(order=True)

    assert edition(all_min_costs, "", "") == Multiset((0,))
    assert edition(all_min_costs, "ab", "bc") == Multiset(
        (2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4)
    )
    # assert edition(all_min_costs, "abba", "abab") == <... 321 results ...>
    # assert edition(all_min_costs, "alberta", "camera") == <... 19825 results ...>

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
    assert edition(count, "ab", "bc") == 13
    assert edition(count, "abba", "abab") == 321
    assert edition(count, "alberta", "camera") == 19825

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
    assert edition(one_align, "ab", "bc") == (("match", "a", "b"), ("match", "b", "c"))
    assert edition(one_align, "abba", "abab") == (
        ("match", "a", "a"),
        ("match", "b", "b"),
        ("match", "b", "a"),
        ("match", "a", "b"),
    )
    assert edition(one_align, "alberta", "camera") == (
        ("delete", "a"),
        ("match", "l", "c"),
        ("match", "b", "a"),
        ("match", "e", "m"),
        ("match", "r", "e"),
        ("match", "t", "r"),
        ("match", "a", "a"),
    )

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
    # edition(all_aligns, "abba", "abab") == Multiset(<... 321 results ...>)
    # edition(all_aligns, "alberta", "camera") == Multiset(<... 19825 results ...>)

    def all_aligns_bruteforce(word1: str, word2: str) -> None:
        assert len(edition(all_aligns, word1, word2)) == edition(count, word1, word2)

    all_aligns_bruteforce("", "")
    all_aligns_bruteforce("ab", "bc")
    all_aligns_bruteforce("abba", "abab")
    all_aligns_bruteforce("alberta", "camera")

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

    assert edition(min_cost_count, "", "") == Record(cost=0, count=1)
    assert edition(min_cost_count, "ab", "bc") == Record(cost=2, count=2)
    assert edition(min_cost_count, "abba", "abab") == Record(cost=2, count=4)
    assert edition(min_cost_count, "alberta", "camera") == Record(cost=4, count=3)

    def min_cost_count_bruteforce(word1: str, word2: str) -> None:
        res_min_cost = edition(min_cost, word1, word2)
        assert edition(min_cost_count, word1, word2) == Record(
            cost=res_min_cost,
            count=sum(
                1
                for align in edition(all_aligns, word1, word2)
                if cost_of(align) == res_min_cost
            ),
        )

    min_cost_count_bruteforce("", "")
    min_cost_count_bruteforce("ab", "bc")
    min_cost_count_bruteforce("abba", "abab")
    min_cost_count_bruteforce("alberta", "camera")

    # Compute the set of alignments of minimum cost
    min_cost_aligns = join(cost=min_cost, solutions=all_aligns) | lex("cost")

    assert edition(min_cost_aligns, "", "") == Record(
        cost=0,
        solutions=Multiset(((),)),
    )

    assert edition(min_cost_aligns, "ab", "bc") == Record(
        cost=2,
        solutions=Multiset(
            (
                (("match", "a", "b"), ("match", "b", "c")),
                (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
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

    assert edition(min_cost_aligns, "alberta", "camera") == Record(
        cost=4,
        solutions=Multiset(
            (
                (
                    ("match", "a", "c"),
                    ("match", "l", "a"),
                    ("match", "b", "m"),
                    ("match", "e", "e"),
                    ("match", "r", "r"),
                    ("delete", "t"),
                    ("match", "a", "a"),
                ),
                (
                    ("insert", "c"),
                    ("match", "a", "a"),
                    ("delete", "l"),
                    ("match", "b", "m"),
                    ("match", "e", "e"),
                    ("match", "r", "r"),
                    ("delete", "t"),
                    ("match", "a", "a"),
                ),
                (
                    ("insert", "c"),
                    ("match", "a", "a"),
                    ("match", "l", "m"),
                    ("delete", "b"),
                    ("match", "e", "e"),
                    ("match", "r", "r"),
                    ("delete", "t"),
                    ("match", "a", "a"),
                ),
            )
        ),
    )

    def min_cost_aligns_bruteforce(word1: str, word2: str) -> None:
        res_min_cost = edition(min_cost, word1, word2)
        assert edition(min_cost_aligns, word1, word2) == Record(
            cost=res_min_cost,
            solutions=Multiset(
                align
                for align in edition(all_aligns, word1, word2)
                if cost_of(align) == res_min_cost
            ),
        )

    min_cost_aligns_bruteforce("", "")
    min_cost_aligns_bruteforce("ab", "bc")
    min_cost_aligns_bruteforce("abba", "abab")
    min_cost_aligns_bruteforce("alberta", "camera")

    # Compute the number of alignments of each cost
    all_costs_count = join(cost=min_cost, count=count) | power() | group("cost")

    assert edition(all_costs_count, "", "") == Multiset((Record(cost=0, count=1),))

    assert edition(all_costs_count, "ab", "bc") == Multiset(
        (
            Record(cost=2, count=2),
            Record(cost=3, count=5),
            Record(cost=4, count=6),
        )
    )

    assert edition(all_costs_count, "abba", "abab") == Multiset(
        (
            Record(cost=2, count=4),
            Record(cost=3, count=7),
            Record(cost=4, count=32),
            Record(cost=5, count=43),
            Record(cost=6, count=95),
            Record(cost=7, count=70),
            Record(cost=8, count=70),
        )
    )

    assert edition(all_costs_count, "alberta", "camera") == Multiset(
        (
            Record(cost=4, count=3),
            Record(cost=5, count=20),
            Record(cost=6, count=97),
            Record(cost=7, count=356),
            Record(cost=8, count=1059),
            Record(cost=9, count=2407),
            Record(cost=10, count=4257),
            Record(cost=11, count=5456),
            Record(cost=12, count=4454),
            Record(cost=13, count=1716),
        )
    )

    def all_costs_count_bruteforce(word1: str, word2: str) -> None:
        assert edition(all_costs_count, word1, word2) == Multiset(
            (
                Record(cost=key, count=value)
                for key, value in Counter(
                    cost_of(align) for align in edition(all_aligns, word1, word2)
                ).items()
            )
        )

    all_costs_count_bruteforce("", "")
    all_costs_count_bruteforce("ab", "bc")
    all_costs_count_bruteforce("abba", "abab")
    all_costs_count_bruteforce("alberta", "camera")

    # Compute the Pareto-optimal number of operations of each type
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
    operations = join(changes=min_change, deletes=min_delete, inserts=min_insert)
    par_operations = operations | power() | pareto("*")

    assert edition(par_operations, "", "") == Multiset(
        (Record(changes=0, deletes=0, inserts=0),)
    )

    assert edition(par_operations, "ab", "bc") == Multiset(
        (
            Record(changes=2, deletes=0, inserts=0),
            Record(changes=0, deletes=1, inserts=1),
        )
    )

    assert edition(par_operations, "abba", "abab") == Multiset(
        (
            Record(changes=2, deletes=0, inserts=0),
            Record(changes=0, deletes=1, inserts=1),
        )
    )

    assert edition(par_operations, "alberta", "camera") == Multiset(
        (
            Record(changes=3, deletes=1, inserts=0),
            Record(changes=1, deletes=2, inserts=1),
            Record(changes=0, deletes=3, inserts=2),
        )
    )

    def operations_of(align: Align) -> Record:
        return Record(
            changes=sum(
                1 if kind == "match" and rest[0] != rest[1] else 0
                for kind, *rest in align
            ),
            deletes=sum(1 if kind == "delete" else 0 for kind, *_ in align),
            inserts=sum(1 if kind == "insert" else 0 for kind, *_ in align),
        )

    def operations_lt(lhs: Record, rhs: Record) -> bool:
        return lhs != rhs and all(
            getattr(lhs, key) <= getattr(rhs, key)
            for key in ("changes", "deletes", "inserts")
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

    def par_operations_bruteforce(word1: str, word2: str) -> None:
        vecs = set(operations_of(align) for align in edition(all_aligns, word1, word2))
        assert edition(par_operations, word1, word2) == Multiset(
            vec for vec in vecs if not any(operations_lt(other, vec) for other in vecs)
        )

    par_operations_bruteforce("", "")
    par_operations_bruteforce("ab", "bc")
    par_operations_bruteforce("abba", "abab")
    par_operations_bruteforce("alberta", "camera")

    # Compute the number of solutions for each Pareto-optimal operation count
    par_operations_count = (
        join(operations=operations, count=count) | power() | pareto("operations.*")
    )

    assert edition(par_operations_count, "", "") == Multiset(
        (
            Record(
                operations=Record(changes=0, deletes=0, inserts=0),
                count=1,
            ),
        )
    )

    assert edition(par_operations_count, "ab", "bc") == Multiset(
        (
            Record(
                operations=Record(changes=2, deletes=0, inserts=0),
                count=1,
            ),
            Record(
                operations=Record(changes=0, deletes=1, inserts=1),
                count=1,
            ),
        )
    )

    assert edition(par_operations_count, "abba", "abab") == Multiset(
        (
            Record(
                operations=Record(changes=2, deletes=0, inserts=0),
                count=1,
            ),
            Record(
                operations=Record(changes=0, deletes=1, inserts=1),
                count=3,
            ),
        )
    )

    assert edition(par_operations_count, "alberta", "camera") == Multiset(
        (
            Record(
                operations=Record(changes=3, deletes=1, inserts=0),
                count=1,
            ),
            Record(
                operations=Record(changes=1, deletes=2, inserts=1),
                count=2,
            ),
            Record(
                operations=Record(changes=0, deletes=3, inserts=2),
                count=3,
            ),
        )
    )

    def par_operations_count_bruteforce(word1: str, word2: str) -> None:
        res_aligns = edition(all_aligns, word1, word2)
        vecs = set(operations_of(align) for align in res_aligns)
        assert edition(par_operations_count, word1, word2) == Multiset(
            Record(
                operations=vec,
                count=sum(1 for align in res_aligns if operations_of(align) == vec),
            )
            for vec in vecs
            if not any(operations_lt(other, vec) for other in vecs)
        )

    par_operations_count_bruteforce("", "")
    par_operations_count_bruteforce("ab", "bc")
    par_operations_count_bruteforce("abba", "abab")
    par_operations_count_bruteforce("alberta", "camera")

    # Compute the set of alignments having Pareto-optimal operation counts
    par_operations_aligns = (
        join(operations=operations, solutions=all_aligns)
        | power()
        | pareto("operations.*")
    )

    assert edition(par_operations_aligns, "", "") == Multiset(
        (
            Record(
                operations=Record(changes=0, deletes=0, inserts=0),
                solutions=Multiset(((),)),
            ),
        )
    )

    assert edition(par_operations_aligns, "ab", "bc") == Multiset(
        (
            Record(
                operations=Record(changes=2, deletes=0, inserts=0),
                solutions=Multiset(((("match", "a", "b"), ("match", "b", "c")),)),
            ),
            Record(
                operations=Record(changes=0, deletes=1, inserts=1),
                solutions=Multiset(
                    ((("delete", "a"), ("match", "b", "b"), ("insert", "c")),)
                ),
            ),
        )
    )

    assert edition(par_operations_aligns, "abba", "abab") == Multiset(
        (
            Record(
                operations=Record(changes=2, deletes=0, inserts=0),
                solutions=Multiset(
                    (
                        (
                            ("match", "a", "a"),
                            ("match", "b", "b"),
                            ("match", "b", "a"),
                            ("match", "a", "b"),
                        ),
                    )
                ),
            ),
            Record(
                operations=Record(changes=0, deletes=1, inserts=1),
                solutions=Multiset(
                    (
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
            ),
        )
    )

    assert edition(par_operations_aligns, "alberta", "camera") == Multiset(
        (
            Record(
                operations=Record(changes=3, deletes=1, inserts=0),
                solutions=Multiset(
                    (
                        (
                            ("match", "a", "c"),
                            ("match", "l", "a"),
                            ("match", "b", "m"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                    )
                ),
            ),
            Record(
                operations=Record(changes=1, deletes=2, inserts=1),
                solutions=Multiset(
                    (
                        (
                            ("insert", "c"),
                            ("match", "a", "a"),
                            ("delete", "l"),
                            ("match", "b", "m"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                        (
                            ("insert", "c"),
                            ("match", "a", "a"),
                            ("match", "l", "m"),
                            ("delete", "b"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                    )
                ),
            ),
            Record(
                operations=Record(changes=0, deletes=3, inserts=2),
                solutions=Multiset(
                    (
                        (
                            ("insert", "c"),
                            ("match", "a", "a"),
                            ("insert", "m"),
                            ("delete", "l"),
                            ("delete", "b"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                        (
                            ("insert", "c"),
                            ("match", "a", "a"),
                            ("delete", "l"),
                            ("insert", "m"),
                            ("delete", "b"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                        (
                            ("insert", "c"),
                            ("match", "a", "a"),
                            ("delete", "l"),
                            ("delete", "b"),
                            ("insert", "m"),
                            ("match", "e", "e"),
                            ("match", "r", "r"),
                            ("delete", "t"),
                            ("match", "a", "a"),
                        ),
                    )
                ),
            ),
        )
    )

    def par_operations_aligns_bruteforce(word1: str, word2: str) -> None:
        res_aligns = edition(all_aligns, word1, word2)
        vecs = set(operations_of(align) for align in res_aligns)
        assert edition(par_operations_aligns, word1, word2) == Multiset(
            Record(
                operations=vec,
                solutions=Multiset(
                    align for align in res_aligns if operations_of(align) == vec
                ),
            )
            for vec in vecs
            if not any(operations_lt(other, vec) for other in vecs)
        )

    par_operations_aligns_bruteforce("", "")
    par_operations_aligns_bruteforce("ab", "bc")
    par_operations_aligns_bruteforce("abba", "abab")
    par_operations_aligns_bruteforce("alberta", "camera")
