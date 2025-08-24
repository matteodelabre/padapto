import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import islice
from math import inf
from typing import Literal

from padapto.algebras import (
    Circuit,
    Signature,
    enumerate_solutions,
    get_solution,
    group,
    join,
    lex,
    limit,
    pareto,
    power,
    trace,
)
from padapto.algebras import (
    make_circuit_node as node,
)
from padapto.collections import Multiset, Record


@dataclass(frozen=True)
class EditionSignature[T](Signature[T]):
    unit: Callable[[], T]
    match: Callable[[T, str, str], T]
    delete: Callable[[T, str], T]
    insert: Callable[[T, str], T]


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
                change = alg.match(table[(i - 1, j - 1)], word1[i - 1], word2[j - 1])

            if i >= 1:
                delete = alg.delete(table[(i - 1, j)], word1[i - 1])

            if j >= 1:
                insert = alg.insert(table[(i, j - 1)], word2[j - 1])

            table[(i, j)] = alg.multichoose(change, delete, insert)

    return table[(n, m)]


# Compute the minimum cost of an alignment, using unit costs
min_cost = EditionSignature[int | float](
    null=lambda: inf,
    choose=min,
    unit=lambda: 0,
    match=lambda cost, sym1, sym2: cost + 1 if sym1 != sym2 else cost,
    delete=lambda cost, sym: cost + 1,
    insert=lambda cost, sym: cost + 1,
)

type Align = tuple[
    tuple[Literal["match"], str, str]
    | tuple[Literal["delete"], str]
    | tuple[Literal["insert"], str],
    ...,
]


def cost_of(align: Align) -> int:
    return sum(
        0 if kind == "match" and rest[0] == rest[1] else 1 for kind, *rest in align
    )


if __name__ == "__main__":
    assert edition(min_cost, "", "") == 0
    assert edition(min_cost, "ab", "bc") == 2
    assert edition(min_cost, "abba", "abab") == 2
    assert edition(min_cost, "alberta", "camera") == 4

    assert cost_of((("match", "a", "a"), ("match", "b", "b"))) == 0
    assert cost_of((("match", "a", "a"), ("match", "b", "c"))) == 1
    assert cost_of((("delete", "a"), ("match", "b", "c"))) == 2
    assert cost_of((("delete", "a"), ("insert", "a"), ("match", "b", "b"))) == 2


# Compute the cost of all alignments, in increasing order
all_min_costs = min_cost | power(order=True)

if __name__ == "__main__":
    assert edition(all_min_costs, "", "") == Multiset((0,))
    assert edition(all_min_costs, "ab", "bc") == Multiset(
        (2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4)
    )
    # assert edition(all_min_costs, "abba", "abab") == <... 321 results ...>
    # assert edition(all_min_costs, "alberta", "camera") == <... 19825 results ...>


# Compute the cost of the 5 best alignments
five_best_costs = min_cost | power(order=True) | limit(5)

if __name__ == "__main__":
    assert edition(five_best_costs, "ab", "bc") == Multiset((2, 2, 3, 3, 3))


# Count the number of possible alignments of two sequences
# See: Laquer H. Turner, (1981), Asymptotic Limits for a Two-Dimensional Recursion
# See: OEIS entry A001850
count = EditionSignature[int](
    null=lambda: 0,
    choose=operator.add,
    unit=lambda: 1,
    match=lambda count, sym1, sym2: count,
    delete=lambda count, sym: count,
    insert=lambda count, sym: count,
)

if __name__ == "__main__":
    assert edition(count, "", "") == 1
    assert edition(count, "ab", "bc") == 13
    assert edition(count, "abba", "abab") == 321
    assert edition(count, "alberta", "camera") == 19825


# Generate circuits representing the possible alignments
trace_align = trace(EditionSignature)


if __name__ == "__main__":
    choose = node("choose")
    unit = node("unit")

    assert edition(trace_align, "", "") == unit

    circ_ab_bc_21 = node("insert", ("b",)).add(unit)
    circ_ab_bc_10 = node("delete", ("a",)).add(unit)
    circ_ab_bc_11 = (
        choose.add(node("match", ("a", "b")).add(unit))
        .add(node("delete", ("a",)).add(circ_ab_bc_21))
        .add(node("insert", ("b",)).add(circ_ab_bc_10))
    )

    assert edition(trace_align, "ab", "bc") == (
        choose.add(node("match", ("b", "c")).add(circ_ab_bc_11))
        .add(
            node("delete", ("b",)).add(
                choose.add(node("match", ("a", "c")).add(circ_ab_bc_21))
                .add(
                    node("delete", ("a",)).add(
                        node("insert", ("c",)).add(circ_ab_bc_21)
                    )
                )
                .add(node("insert", ("c",)).add(circ_ab_bc_11))
            )
        )
        .add(
            node("insert", ("c",)).add(
                choose.add(node("match", ("b", "b")).add(circ_ab_bc_10))
                .add(node("delete", ("b",)).add(circ_ab_bc_11))
                .add(
                    node("insert", ("b",)).add(
                        node("delete", ("b",)).add(circ_ab_bc_10)
                    )
                )
            )
        )
    )


# Enumerate solutions from the generated circuits
def flatten_align(solution: Circuit) -> Align:
    if not solution.edges:
        return ()

    operation = (solution.data.operator, *solution.data.args)
    return flatten_align(solution.edges[0].node) + (operation,)


def one_align(circuit: Circuit) -> Align:
    return flatten_align(get_solution(circuit))


def all_aligns(circuit: Circuit) -> Iterable[Align]:
    for solution in enumerate_solutions(circuit):
        yield flatten_align(solution)


def iterables_equal[T](left: Iterable[T], right: Iterable[T]) -> bool:
    try:
        for lhs, rhs in zip(left, right, strict=True):
            if lhs != rhs:
                return False
    except ValueError:
        # Iterables have different lengths
        return False

    return True


if __name__ == "__main__":
    assert one_align(edition(trace_align, "", "")) == ()
    assert iterables_equal(
        all_aligns(edition(trace_align, "", "")),
        ((),),
    )

    assert one_align(edition(trace_align, "ab", "bc")) == (
        ("match", "a", "b"),
        ("match", "b", "c"),
    )
    assert iterables_equal(
        all_aligns(edition(trace_align, "ab", "bc")),
        (
            (("match", "a", "b"), ("match", "b", "c")),
            (("insert", "b"), ("delete", "a"), ("match", "b", "c")),
            (("delete", "a"), ("insert", "b"), ("match", "b", "c")),
            (("insert", "b"), ("match", "a", "c"), ("delete", "b")),
            (("insert", "b"), ("insert", "c"), ("delete", "a"), ("delete", "b")),
            (("match", "a", "b"), ("insert", "c"), ("delete", "b")),
            (("insert", "b"), ("delete", "a"), ("insert", "c"), ("delete", "b")),
            (("delete", "a"), ("insert", "b"), ("insert", "c"), ("delete", "b")),
            (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
            (("match", "a", "b"), ("delete", "b"), ("insert", "c")),
            (("insert", "b"), ("delete", "a"), ("delete", "b"), ("insert", "c")),
            (("delete", "a"), ("insert", "b"), ("delete", "b"), ("insert", "c")),
            (("delete", "a"), ("delete", "b"), ("insert", "b"), ("insert", "c")),
        ),
    )

    assert one_align(edition(trace_align, "abba", "abab")) == (
        ("match", "a", "a"),
        ("match", "b", "b"),
        ("match", "b", "a"),
        ("match", "a", "b"),
    )
    # Retrieving all aligns on this example would yield 321 results

    assert one_align(edition(trace_align, "alberta", "camera")) == (
        ("delete", "a"),
        ("match", "l", "c"),
        ("match", "b", "a"),
        ("match", "e", "m"),
        ("match", "r", "e"),
        ("match", "t", "r"),
        ("match", "a", "a"),
    )
    # Retrieving all aligns on this example would yield 19825 results

    def all_aligns_bruteforce(word1: str, word2: str) -> None:
        total = sum(1 for _ in all_aligns(edition(trace_align, word1, word2)))
        assert total == edition(count, word1, word2)

    all_aligns_bruteforce("", "")
    all_aligns_bruteforce("ab", "bc")
    all_aligns_bruteforce("abba", "abab")
    all_aligns_bruteforce("alberta", "camera")

    # For n=15, there are 44,642,381,823 solutions; the following will only finish if
    # `enumerate_candidates` generates its solutions lazily
    circuit = edition(trace_align, "a" * 15, "a" * 15)
    sols = list(islice(enumerate_solutions(circuit), 100))

    # Check that the minimum cost value is correct
    assert edition(min_cost, "abba", "abab") == min(
        cost_of(align) for align in all_aligns(edition(trace_align, "abba", "abab"))
    )
    assert edition(all_min_costs, "abba", "abab") == Multiset(
        cost_of(align) for align in all_aligns(edition(trace_align, "abba", "abab"))
    )


# Compute the number of alignments of minimum cost
min_cost_count = join(cost=min_cost, count=count) | lex("cost")

if __name__ == "__main__":
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
                for align in all_aligns(edition(trace_align, word1, word2))
                if cost_of(align) == res_min_cost
            ),
        )

    min_cost_count_bruteforce("", "")
    min_cost_count_bruteforce("ab", "bc")
    min_cost_count_bruteforce("abba", "abab")
    min_cost_count_bruteforce("alberta", "camera")


# Compute the set of alignments of minimum cost
min_cost_aligns = join(cost=min_cost, solutions=trace_align) | lex("cost")

if __name__ == "__main__":
    assert edition(min_cost_aligns, "", "").cost == 0
    assert iterables_equal(
        all_aligns(edition(min_cost_aligns, "", "").solutions),
        ((),),
    )

    assert edition(min_cost_aligns, "ab", "bc").cost == 2
    assert iterables_equal(
        all_aligns(edition(min_cost_aligns, "ab", "bc").solutions),
        (
            (("match", "a", "b"), ("match", "b", "c")),
            (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
        ),
    )

    assert edition(min_cost_aligns, "abba", "abab").cost == 2
    assert iterables_equal(
        all_aligns(edition(min_cost_aligns, "abba", "abab").solutions),
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
        ),
    )

    assert edition(min_cost_aligns, "alberta", "camera").cost == 4
    assert iterables_equal(
        all_aligns(edition(min_cost_aligns, "alberta", "camera").solutions),
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
        ),
    )

    def min_cost_aligns_bruteforce(word1: str, word2: str) -> None:
        result = edition(min_cost_aligns, word1, word2)
        assert result.cost == edition(min_cost, word1, word2)
        assert iterables_equal(
            all_aligns(result.solutions),
            (
                align
                for align in all_aligns(edition(trace_align, word1, word2))
                if cost_of(align) == result.cost
            ),
        )

    min_cost_aligns_bruteforce("", "")
    min_cost_aligns_bruteforce("ab", "bc")
    min_cost_aligns_bruteforce("abba", "abab")
    min_cost_aligns_bruteforce("alberta", "camera")


# Compute the number of alignments of each cost
all_costs_count = join(cost=min_cost, count=count) | power() | group("cost")

if __name__ == "__main__":
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
                    cost_of(align)
                    for align in all_aligns(edition(trace_align, word1, word2))
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
    match=lambda count, sym1, sym2: count + 1 if sym1 != sym2 else count,
    delete=lambda count, sym: count,
    insert=lambda count, sym: count,
)
min_delete = replace(
    min_cost,
    match=lambda count, sym1, sym2: count,
    delete=lambda count, sym: count + 1,
    insert=lambda count, sym: count,
)
min_insert = replace(
    min_cost,
    match=lambda count, sym1, sym2: count,
    delete=lambda count, sym: count,
    insert=lambda count, sym: count + 1,
)
operations = join(changes=min_change, deletes=min_delete, inserts=min_insert)
par_operations = operations | power() | pareto("*")

if __name__ == "__main__":
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
        vecs = set(
            operations_of(align)
            for align in all_aligns(edition(trace_align, word1, word2))
        )
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

if __name__ == "__main__":
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
        res_aligns = tuple(all_aligns(edition(trace_align, word1, word2)))
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
    join(operations=operations, solutions=trace_align)
    | power()
    | pareto("operations.*")
)

if __name__ == "__main__":
    empty_aligns = edition(par_operations_aligns, "", "")
    assert empty_aligns[0].operations == Record(changes=0, deletes=0, inserts=0)
    assert iterables_equal(
        all_aligns(empty_aligns[0].solutions),
        ((),),
    )

    ab_bc_aligns = edition(par_operations_aligns, "ab", "bc")
    assert ab_bc_aligns[0].operations == Record(changes=2, deletes=0, inserts=0)
    assert iterables_equal(
        all_aligns(ab_bc_aligns[0].solutions),
        ((("match", "a", "b"), ("match", "b", "c")),),
    )
    assert ab_bc_aligns[1].operations == Record(changes=0, deletes=1, inserts=1)
    assert iterables_equal(
        all_aligns(ab_bc_aligns[1].solutions),
        ((("delete", "a"), ("match", "b", "b"), ("insert", "c")),),
    )

    abba_abab_aligns = edition(par_operations_aligns, "abba", "abab")
    assert abba_abab_aligns[0].operations == Record(changes=2, deletes=0, inserts=0)
    assert iterables_equal(
        all_aligns(abba_abab_aligns[0].solutions),
        (
            (
                ("match", "a", "a"),
                ("match", "b", "b"),
                ("match", "b", "a"),
                ("match", "a", "b"),
            ),
        ),
    )
    assert abba_abab_aligns[1].operations == Record(changes=0, deletes=1, inserts=1)
    assert iterables_equal(
        all_aligns(abba_abab_aligns[1].solutions),
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
        ),
    )

    alberta_camera_aligns = edition(par_operations_aligns, "alberta", "camera")
    assert alberta_camera_aligns[0].operations == Record(
        changes=3, deletes=1, inserts=0
    )
    assert iterables_equal(
        all_aligns(alberta_camera_aligns[0].solutions),
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
        ),
    )
    assert alberta_camera_aligns[1].operations == Record(
        changes=1, deletes=2, inserts=1
    )
    assert iterables_equal(
        all_aligns(alberta_camera_aligns[1].solutions),
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
        ),
    )
    assert alberta_camera_aligns[2].operations == Record(
        changes=0, deletes=3, inserts=2
    )
    assert iterables_equal(
        all_aligns(alberta_camera_aligns[2].solutions),
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
        ),
    )

    def par_operations_aligns_bruteforce(word1: str, word2: str) -> None:
        result = edition(par_operations_aligns, word1, word2)

        for item in result:
            assert iterables_equal(
                all_aligns(item.solutions),
                (
                    align
                    for align in all_aligns(edition(trace_align, word1, word2))
                    if operations_of(align) == item.operations
                ),
            )

    par_operations_aligns_bruteforce("", "")
    par_operations_aligns_bruteforce("ab", "bc")
    par_operations_aligns_bruteforce("abba", "abab")
    par_operations_aligns_bruteforce("alberta", "camera")
