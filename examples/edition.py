import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice
from random import Random
from typing import Literal, cast

from padapto.algebras import (
    Operator,
    Signature,
    add_optimizer,
    boltzmann,
    counter,
    group,
    join,
    lex,
    limit,
    pareto,
    power,
    trace,
)
from padapto.circuit import (
    Circuit,
    enumerate_solutions,
    get_solution,
    make_node,
    render,
    sample,
)
from padapto.collections import Multiset, Record
from padapto.structure import (
    Empty,
    Grammar,
    Item,
    Subseq,
    Var,
    chain,
    clause,
    grammar,
    predicate,
)

from .gettotalsize import gettotalsize


@dataclass(frozen=True)
class AlignSignature[T](Signature[T]):
    unit: Callable[[], T]
    match: Callable[[T, str, str], T]
    delete: Callable[[T, str], T]
    insert: Callable[[T, str], T]


@grammar
class AlignGrammar[T](Grammar[T]):
    alg: AlignSignature[T]

    @predicate
    @staticmethod
    def align(left: str, right: str) -> T:
        return  # type: ignore

    @clause(left=Empty(), right=Empty())
    def _empty(self):
        return self.alg.unit()

    @clause(
        left=chain(Item(Var("left_char")), Subseq(Var("left"))),
        right=chain(Item(Var("right_char")), Subseq(Var("right"))),
    )
    def _match(self, left_char: str, right_char: str, left: str, right: str) -> T:
        return self.alg.match(self.align(left=left, right=right), left_char, right_char)

    @clause(
        left=chain(Item(Var("left_char")), Subseq(Var("left"))),
        right=Var("right"),
    )
    def _delete(self, left_char: str, left: str, right: str) -> T:
        return self.alg.delete(self.align(left=left, right=right), left_char)

    @clause(
        left=Var("left"),
        right=chain(Item(Var("right_char")), Subseq(Var("right"))),
    )
    def _insert(self, right_char: str, left: str, right: str) -> T:
        return self.alg.insert(self.align(left=left, right=right), right_char)


# Compute the minimum cost of an alignment, using unit costs
def _unit_cost_match(sym1: str, sym2: str) -> int:
    return 1 if sym1 != sym2 else 0


def _unit_cost_delete(sym: str) -> int:
    return 1


def _unit_cost_insert(sym: str) -> int:
    return 1


unit_cost_ops: dict[str, Operator[float]] = {
    "match": _unit_cost_match,
    "delete": _unit_cost_delete,
    "insert": _unit_cost_insert,
}
min_cost: AlignSignature[float] = add_optimizer(
    AlignSignature, choose="min", **unit_cost_ops
)
boltz_distr: AlignSignature[float] = boltzmann(
    AlignSignature, temperature=1, **unit_cost_ops
)
gr_min_cost = AlignGrammar(min_cost).align


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
    assert gr_min_cost(left="", right="") == 0
    assert gr_min_cost(left="ab", right="bc") == 2
    assert gr_min_cost(left="abba", right="abab") == 2
    assert gr_min_cost(left="alberta", right="camera") == 4

    assert cost_of((("match", "a", "a"), ("match", "b", "b"))) == 0
    assert cost_of((("match", "a", "a"), ("match", "b", "c"))) == 1
    assert cost_of((("delete", "a"), ("match", "b", "c"))) == 2
    assert cost_of((("delete", "a"), ("insert", "a"), ("match", "b", "b"))) == 2


# Compute the cost of all alignments, in increasing order
all_min_costs = min_cost | power(order=True)
gr_all_min_costs = AlignGrammar(all_min_costs).align

if __name__ == "__main__":
    assert gr_all_min_costs(left="", right="") == Multiset((0,))
    assert gr_all_min_costs(left="ab", right="bc") == Multiset(
        (2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4)
    )
    # assert gr_all_min_costs(left="abba", right="abab") == <... 321 results ...>
    # assert gr_all_min_costs(left="alberta", right="camera") == <... 19825 results ...>


# Compute the cost of the 5 best alignments
five_best_costs = min_cost | power(order=True) | limit(5)
gr_five_best_costs = AlignGrammar(five_best_costs).align

if __name__ == "__main__":
    assert gr_five_best_costs(left="ab", right="bc") == Multiset((2, 2, 3, 3, 3))


# Count the number of possible alignments of two sequences
# See: Laquer H. Turner, (1981), Asymptotic Limits for a Two-Dimensional Recursion
# See: OEIS entry A001850
count: AlignSignature[int] = counter(AlignSignature)
gr_count = AlignGrammar(count).align

if __name__ == "__main__":
    assert gr_count(left="", right="") == 1
    assert gr_count(left="ab", right="bc") == 13
    assert gr_count(left="abba", right="abab") == 321
    assert gr_count(left="alberta", right="camera") == 19825


# Generate circuits representing the possible alignments
tracer = trace(AlignSignature)
gr_tracer = AlignGrammar(tracer).align

if __name__ == "__main__":
    choose = make_node("choose")
    unit = make_node("unit")

    assert gr_tracer(left="", right="") == unit

    circ_ab_bc_21 = make_node("insert", ("b",)).add(unit)
    circ_ab_bc_10 = make_node("delete", ("a",)).add(unit)
    circ_ab_bc_11 = (
        choose.add(make_node("match", ("a", "b")).add(unit))
        .add(make_node("delete", ("a",)).add(circ_ab_bc_21))
        .add(make_node("insert", ("b",)).add(circ_ab_bc_10))
    )

    assert gr_tracer(left="ba", right="cb") == (
        choose.add(make_node("match", ("b", "c")).add(circ_ab_bc_11))
        .add(
            make_node("delete", ("b",)).add(
                choose.add(make_node("match", ("a", "c")).add(circ_ab_bc_21))
                .add(
                    make_node("delete", ("a",)).add(
                        make_node("insert", ("c",)).add(circ_ab_bc_21)
                    )
                )
                .add(make_node("insert", ("c",)).add(circ_ab_bc_11))
            )
        )
        .add(
            make_node("insert", ("c",)).add(
                choose.add(make_node("match", ("b", "b")).add(circ_ab_bc_10))
                .add(make_node("delete", ("b",)).add(circ_ab_bc_11))
                .add(
                    make_node("insert", ("b",)).add(
                        make_node("delete", ("b",)).add(circ_ab_bc_10)
                    )
                )
            )
        )
    )

    assert render(gr_tracer(left="ba", right="cb")) == """\
digraph {
0 [label="⊕", shape="none", width="0", height="0"]
0 -> 1
0 -> 2
0 -> 3
1 [ordering="out", shape="box", style="rounded", label="match('b', 'c')"]
1 -> 4
4 [label="⊕", shape="none", width="0", height="0"]
4 -> 5
4 -> 6
4 -> 7
5 [ordering="out", shape="box", style="rounded", label="match('a', 'b')"]
5 -> 8
8 [ordering="out", shape="box", style="rounded", label="unit"]
6 [ordering="out", shape="box", style="rounded", label="delete('a')"]
6 -> 9
9 [ordering="out", shape="box", style="rounded", label="insert('b')"]
9 -> 8
7 [ordering="out", shape="box", style="rounded", label="insert('b')"]
7 -> 10
10 [ordering="out", shape="box", style="rounded", label="delete('a')"]
10 -> 8
2 [ordering="out", shape="box", style="rounded", label="delete('b')"]
2 -> 11
11 [label="⊕", shape="none", width="0", height="0"]
11 -> 12
11 -> 13
11 -> 14
12 [ordering="out", shape="box", style="rounded", label="match('a', 'c')"]
12 -> 9
13 [ordering="out", shape="box", style="rounded", label="delete('a')"]
13 -> 15
15 [ordering="out", shape="box", style="rounded", label="insert('c')"]
15 -> 9
14 [ordering="out", shape="box", style="rounded", label="insert('c')"]
14 -> 4
3 [ordering="out", shape="box", style="rounded", label="insert('c')"]
3 -> 16
16 [label="⊕", shape="none", width="0", height="0"]
16 -> 17
16 -> 18
16 -> 19
17 [ordering="out", shape="box", style="rounded", label="match('b', 'b')"]
17 -> 10
18 [ordering="out", shape="box", style="rounded", label="delete('b')"]
18 -> 4
19 [ordering="out", shape="box", style="rounded", label="insert('b')"]
19 -> 20
20 [ordering="out", shape="box", style="rounded", label="delete('b')"]
20 -> 10
}\
"""


# Enumerate solutions from the generated circuits
def flatten_align(solution: Circuit) -> Align:
    if not solution.edges:
        return ()

    operation = (solution.data.operator, *solution.data.args)
    return (operation,) + flatten_align(solution.edges[0].node)


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
    assert one_align(gr_tracer(left="", right="")) == ()
    assert iterables_equal(
        all_aligns(gr_tracer(left="", right="")),
        ((),),
    )

    assert one_align(gr_tracer(left="ab", right="bc")) == (
        ("match", "a", "b"),
        ("match", "b", "c"),
    )
    assert iterables_equal(
        all_aligns(gr_tracer(left="ab", right="bc")),
        (
            (("match", "a", "b"), ("match", "b", "c")),
            (("match", "a", "b"), ("delete", "b"), ("insert", "c")),
            (("match", "a", "b"), ("insert", "c"), ("delete", "b")),
            (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
            (("delete", "a"), ("delete", "b"), ("insert", "b"), ("insert", "c")),
            (("delete", "a"), ("insert", "b"), ("match", "b", "c")),
            (("delete", "a"), ("insert", "b"), ("delete", "b"), ("insert", "c")),
            (("delete", "a"), ("insert", "b"), ("insert", "c"), ("delete", "b")),
            (("insert", "b"), ("match", "a", "c"), ("delete", "b")),
            (("insert", "b"), ("delete", "a"), ("match", "b", "c")),
            (("insert", "b"), ("delete", "a"), ("delete", "b"), ("insert", "c")),
            (("insert", "b"), ("delete", "a"), ("insert", "c"), ("delete", "b")),
            (("insert", "b"), ("insert", "c"), ("delete", "a"), ("delete", "b")),
        ),
    )

    assert one_align(gr_tracer(left="abba", right="abab")) == (
        ("match", "a", "a"),
        ("match", "b", "b"),
        ("match", "b", "a"),
        ("match", "a", "b"),
    )
    # Retrieving all aligns on this example would yield 321 results

    assert one_align(gr_tracer(left="alberta", right="camera")) == (
        ("match", "a", "c"),
        ("match", "l", "a"),
        ("match", "b", "m"),
        ("match", "e", "e"),
        ("match", "r", "r"),
        ("match", "t", "a"),
        ("delete", "a"),
    )
    # Retrieving all aligns on this example would yield 19825 results

    def all_aligns_bruteforce(word1: str, word2: str) -> None:
        total = sum(1 for _ in all_aligns(gr_tracer(left=word1, right=word2)))
        assert total == gr_count(left=word1, right=word2)

    all_aligns_bruteforce("", "")
    all_aligns_bruteforce("ab", "bc")
    all_aligns_bruteforce("abba", "abab")
    all_aligns_bruteforce("alberta", "camera")

    # For n=15, there are 44,642,381,823 solutions; the following will only finish if
    # `enumerate_candidates` generates its solutions lazily
    circuit = gr_tracer(left="a" * 15, right="a" * 15)
    sols = list(islice(enumerate_solutions(circuit), 100))

    # Check that the minimum cost value is correct
    assert gr_min_cost(left="abba", right="abab") == min(
        cost_of(align) for align in all_aligns(gr_tracer(left="abba", right="abab"))
    )
    assert gr_all_min_costs(left="abba", right="abab") == Multiset(
        cost_of(align) for align in all_aligns(gr_tracer(left="abba", right="abab"))
    )

    # Sample a solution from a Boltzmann distribution
    gen = Random(42)
    circ = gr_tracer(left="alberta", right="camera")
    min_cost_circ = 2 * gr_min_cost(left="alberta", right="camera")

    sampled_boltz_sol = sample(circ, gen, boltz_distr)
    assert sampled_boltz_sol in enumerate_solutions(circ)
    assert cost_of(flatten_align(sampled_boltz_sol)) <= min_cost_circ


# Compute the number of alignments of minimum cost
min_cost_count = join(cost=min_cost, count=count) | lex("cost")
gr_min_cost_count = AlignGrammar(min_cost_count).align

if __name__ == "__main__":
    assert gr_min_cost_count(left="", right="") == Record(cost=0, count=1)
    assert gr_min_cost_count(left="ab", right="bc") == Record(cost=2, count=2)
    assert gr_min_cost_count(left="abba", right="abab") == Record(cost=2, count=4)
    assert gr_min_cost_count(left="alberta", right="camera") == Record(cost=4, count=3)

    def min_cost_count_bruteforce(word1: str, word2: str) -> None:
        res_min_cost = gr_min_cost(left=word1, right=word2)
        assert gr_min_cost_count(left=word1, right=word2) == Record(
            cost=res_min_cost,
            count=sum(
                1
                for align in all_aligns(gr_tracer(left=word1, right=word2))
                if cost_of(align) == res_min_cost
            ),
        )

    min_cost_count_bruteforce("", "")
    min_cost_count_bruteforce("ab", "bc")
    min_cost_count_bruteforce("abba", "abab")
    min_cost_count_bruteforce("alberta", "camera")


# Compute the set of alignments of minimum cost
min_cost_aligns = join(cost=min_cost, solutions=tracer) | lex("cost")
gr_min_cost_aligns = AlignGrammar(min_cost_aligns).align

if __name__ == "__main__":
    assert gr_min_cost_aligns(left="", right="").cost == 0
    assert iterables_equal(
        all_aligns(gr_min_cost_aligns(left="", right="").solutions),
        ((),),
    )

    assert gr_min_cost_aligns(left="ab", right="bc").cost == 2
    assert iterables_equal(
        all_aligns(gr_min_cost_aligns(left="ab", right="bc").solutions),
        (
            (("match", "a", "b"), ("match", "b", "c")),
            (("delete", "a"), ("match", "b", "b"), ("insert", "c")),
        ),
    )

    assert gr_min_cost_aligns(left="abba", right="abab").cost == 2
    assert iterables_equal(
        all_aligns(gr_min_cost_aligns(left="abba", right="abab").solutions),
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
                ("delete", "b"),
                ("match", "a", "a"),
                ("insert", "b"),
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
        ),
    )

    assert gr_min_cost_aligns(left="alberta", right="camera").cost == 4
    assert iterables_equal(
        all_aligns(gr_min_cost_aligns(left="alberta", right="camera").solutions),
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
                ("match", "l", "m"),
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
                ("match", "b", "m"),
                ("match", "e", "e"),
                ("match", "r", "r"),
                ("delete", "t"),
                ("match", "a", "a"),
            ),
        ),
    )

    def min_cost_aligns_bruteforce(word1: str, word2: str) -> None:
        result = gr_min_cost_aligns(left=word1, right=word2)
        assert result.cost == gr_min_cost(left=word1, right=word2)
        assert iterables_equal(
            all_aligns(result.solutions),
            (
                align
                for align in all_aligns(gr_tracer(left=word1, right=word2))
                if cost_of(align) == result.cost
            ),
        )

    min_cost_aligns_bruteforce("", "")
    min_cost_aligns_bruteforce("ab", "bc")
    min_cost_aligns_bruteforce("abba", "abab")
    min_cost_aligns_bruteforce("alberta", "camera")

    # Randomly sample among optimal solutions
    gen = Random(42)
    circ = gr_min_cost_aligns(left="alberta", right="camera").solutions
    sampled_min_sol = sample(circ, gen, cast(AlignSignature[float], count))
    assert sampled_min_sol in enumerate_solutions(circ)


# Compute the number of alignments of each cost
all_costs_count = join(cost=min_cost, count=count) | power() | group("cost")
gr_all_costs_count = AlignGrammar(all_costs_count).align

if __name__ == "__main__":
    assert gr_all_costs_count(left="", right="") == Multiset((Record(cost=0, count=1),))

    assert gr_all_costs_count(left="ab", right="bc") == Multiset(
        (
            Record(cost=2, count=2),
            Record(cost=3, count=5),
            Record(cost=4, count=6),
        )
    )

    assert gr_all_costs_count(left="abba", right="abab") == Multiset(
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

    assert gr_all_costs_count(left="alberta", right="camera") == Multiset(
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
        assert gr_all_costs_count(left=word1, right=word2) == Multiset(
            (
                Record(cost=key, count=value)
                for key, value in Counter(
                    cost_of(align)
                    for align in all_aligns(gr_tracer(left=word1, right=word2))
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
gr_par_operations = AlignGrammar(par_operations).align

if __name__ == "__main__":
    assert gr_par_operations(left="", right="") == Multiset(
        (Record(changes=0, deletes=0, inserts=0),)
    )

    assert gr_par_operations(left="ab", right="bc") == Multiset(
        (
            Record(changes=2, deletes=0, inserts=0),
            Record(changes=0, deletes=1, inserts=1),
        )
    )

    assert gr_par_operations(left="abba", right="abab") == Multiset(
        (
            Record(changes=2, deletes=0, inserts=0),
            Record(changes=0, deletes=1, inserts=1),
        )
    )

    assert gr_par_operations(left="alberta", right="camera") == Multiset(
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
            for align in all_aligns(gr_tracer(left=word1, right=word2))
        )
        assert gr_par_operations(left=word1, right=word2) == Multiset(
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
gr_par_operations_count = AlignGrammar(par_operations_count).align

if __name__ == "__main__":
    assert gr_par_operations_count(left="", right="") == Multiset(
        (
            Record(
                operations=Record(changes=0, deletes=0, inserts=0),
                count=1,
            ),
        )
    )

    assert gr_par_operations_count(left="ab", right="bc") == Multiset(
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

    assert gr_par_operations_count(left="abba", right="abab") == Multiset(
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

    assert gr_par_operations_count(left="alberta", right="camera") == Multiset(
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
        res_aligns = tuple(all_aligns(gr_tracer(left=word1, right=word2)))
        vecs = set(operations_of(align) for align in res_aligns)
        assert gr_par_operations_count(left=word1, right=word2) == Multiset(
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
    join(operations=operations, solutions=tracer) | power() | pareto("operations.*")
)
gr_par_operations_aligns = AlignGrammar(par_operations_aligns).align

if __name__ == "__main__":
    empty_aligns = gr_par_operations_aligns(left="", right="")
    assert empty_aligns[0].operations == Record(changes=0, deletes=0, inserts=0)
    assert iterables_equal(
        all_aligns(empty_aligns[0].solutions),
        ((),),
    )

    ab_bc_aligns = gr_par_operations_aligns(left="ab", right="bc")
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

    abba_abab_aligns = gr_par_operations_aligns(left="abba", right="abab")
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
                ("delete", "b"),
                ("match", "a", "a"),
                ("insert", "b"),
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
        ),
    )

    alberta_camera_aligns = gr_par_operations_aligns(left="alberta", right="camera")
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
                ("match", "l", "m"),
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
                ("match", "b", "m"),
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
                ("delete", "l"),
                ("delete", "b"),
                ("insert", "m"),
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
                ("insert", "m"),
                ("delete", "l"),
                ("delete", "b"),
                ("match", "e", "e"),
                ("match", "r", "r"),
                ("delete", "t"),
                ("match", "a", "a"),
            ),
        ),
    )

    def par_operations_aligns_bruteforce(word1: str, word2: str) -> None:
        result = gr_par_operations_aligns(left=word1, right=word2)

        for item in result:
            assert iterables_equal(
                all_aligns(item.solutions),
                (
                    align
                    for align in all_aligns(gr_tracer(left=word1, right=word2))
                    if operations_of(align) == item.operations
                ),
            )

    par_operations_aligns_bruteforce("", "")
    par_operations_aligns_bruteforce("ab", "bc")
    par_operations_aligns_bruteforce("abba", "abab")
    par_operations_aligns_bruteforce("alberta", "camera")


# Performance test
if __name__ == "__main__":
    print("Running performance test")
    print('"n","Memory (kB)","Time (ms)"')

    for n in range(1, 101):
        data = {"left": "a" * n, "right": "a" * n}
        gr_perf_test = AlignGrammar(min_cost)

        start = time.perf_counter_ns()
        res = gr_perf_test.align(**data)
        end = time.perf_counter_ns()

        print(
            n,
            gettotalsize(gr_perf_test) // 1_000,
            (end - start) // 1_000_000,
            sep=",",
        )
