from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import Random
from typing import Literal, cast

from sowing import Hedge, Node
from sowing.repr.newick import parse

from padapto.algebras import Signature, add_optimizer, boltzmann, counter, trace
from padapto.circuit import Circuit, sample
from padapto.structure import (
    Empty,
    Grammar,
    Item,
    Subseq,
    Tree,
    Var,
    chain,
    clause,
    grammar,
    predicate,
)

StrHedge = Hedge[Mapping[str, str], None]


@dataclass(frozen=True)
class ForestSignature[T](Signature[T]):
    unit: Callable[[], T]
    match: Callable[[T, T, str, str], T]
    insert: Callable[[T, T, str], T]
    delete: Callable[[T, T, str], T]


@grammar
class ForestGrammar[T](Grammar[T]):
    alg: ForestSignature[T]

    @predicate
    @staticmethod
    def align(left: StrHedge, right: StrHedge) -> T:
        return  # type: ignore

    @clause(left=Empty(), right=Empty())
    def _empty(self):
        return self.alg.unit()

    @clause(
        left=chain(
            Item(Tree(node=Var("left_data"), children=Var("left_down"))),
            Subseq(Var("left_rest")),
        ),
        right=chain(
            Item(Tree(node=Var("right_data"), children=Var("right_down"))),
            Subseq(Var("right_rest")),
        ),
    )
    def match(
        self,
        left_data: Mapping[str, str],
        left_down: StrHedge,
        left_rest: StrHedge,
        right_data: Mapping[str, str],
        right_down: StrHedge,
        right_rest: StrHedge,
    ) -> T:
        return self.alg.match(
            self.align(left=left_down, right=right_down),
            self.align(left=left_rest, right=right_rest),
            left_data["name"],
            right_data["name"],
        )

    @clause(
        left=chain(Subseq(Var("left_down")), Subseq(Var("left_rest"))),
        right=chain(
            Item(Tree(node=Var("right_data"), children=Var("right_down"))),
            Subseq(Var("right_rest")),
        ),
    )
    def insert(
        self,
        left_down: StrHedge,
        left_rest: StrHedge,
        right_data: Mapping[str, str],
        right_down: StrHedge,
        right_rest: StrHedge,
    ) -> T:
        return self.alg.insert(
            self.align(left=left_down, right=right_down),
            self.align(left=left_rest, right=right_rest),
            right_data["name"],
        )

    @clause(
        left=chain(
            Item(Tree(node=Var("left_data"), children=Var("left_down"))),
            Subseq(Var("left_rest")),
        ),
        right=chain(Subseq(Var("right_down")), Subseq(Var("right_rest"))),
    )
    def delete(
        self,
        left_data: Mapping[str, str],
        left_down: StrHedge,
        left_rest: StrHedge,
        right_down: StrHedge,
        right_rest: StrHedge,
    ) -> T:
        return self.alg.delete(
            self.align(left=left_down, right=right_down),
            self.align(left=left_rest, right=right_rest),
            left_data["name"],
        )


AlignForest = Node[
    (
        tuple[Literal["match"], str, str]
        | tuple[Literal["delete"], str]
        | tuple[Literal["insert"], str]
    ),
    None,
]


def flatten_align(solution: Circuit) -> tuple[AlignForest, ...]:
    if not solution.edges:
        return ()

    operation = (solution.data.operator, *solution.data.args)
    return (
        AlignForest(operation).extend(flatten_align(solution.edges[0].node)),
        *flatten_align(solution.edges[1].node),
    )


# Compute the number of alignments
align_counter = counter(ForestSignature)
gr_counter = ForestGrammar(align_counter).align

if __name__ == "__main__":
    leaf = cast(StrHedge, Hedge.of(parse("a;")))
    assert gr_counter(left=leaf, right=leaf) == 5


# Generate and sample solutions of minimum cost
align_tracer = trace(ForestSignature)
gr_tracer = ForestGrammar(align_tracer).align


def _unit_cost_match(sym1: str, sym2: str) -> int:
    return 1 if sym1 != sym2 else 0


def _unit_cost_delete(sym: str) -> int:
    return 1


def _unit_cost_insert(sym: str) -> int:
    return 1


align_min_cost = add_optimizer(
    ForestSignature,
    choose="min",
    match=_unit_cost_match,
    insert=_unit_cost_insert,
    delete=_unit_cost_delete,
)
align_boltzmann = boltzmann(
    ForestSignature,
    temperature=0.1,
    match=_unit_cost_match,
    insert=_unit_cost_insert,
    delete=_unit_cost_delete,
)
gr_min_cost = ForestGrammar(align_min_cost).align

if __name__ == "__main__":
    left = cast(StrHedge, Hedge.of(parse("((d,e,f)b,(g,h)c)a;")))
    right = cast(StrHedge, Hedge.of(parse("((d)b,(e,f)i,(g)d)a;")))
    assert gr_min_cost(left=left, right=right) == 5

    circ = gr_tracer(left=left, right=right)
    sol = sample(circ, Random(42), align_boltzmann)
    assert flatten_align(sol) == (
        AlignForest(("match", "a", "a"))
        .add(
            AlignForest(("delete", "b"))
            .add(AlignForest(("insert", "b")).add(AlignForest(("match", "d", "d"))))
            .add(
                AlignForest(("insert", "i"))
                .add(AlignForest(("match", "e", "e")))
                .add(AlignForest(("match", "f", "f")))
            )
        )
        .add(
            AlignForest(("match", "c", "d"))
            .add(AlignForest(("match", "g", "g")))
            .add(AlignForest(("delete", "h")))
        ),
    )
