import operator
from collections.abc import Callable
from dataclasses import dataclass

from immutables import Map
from sowing import Node

from padapto.algebras import Signature
from padapto.structure.grammar import Grammar, clause, grammar, predicate
from padapto.structure.pattern import Empty, Item, Subseq, Var, Zero, chain


@dataclass(frozen=True, slots=True)
class ExampleSignature[Out](Signature[Out]):
    unit: Callable[[], Out]
    comb1: Callable[[int, Out, Out], Out]
    comb2: Callable[[str, Out, Out], Out]


def test_grammar_class() -> None:
    @grammar
    class ExampleGrammar[Out](Grammar[Out]):
        alg: ExampleSignature[Out]

        @predicate
        @staticmethod
        def pred1(t1: str, t2: int, t3: Node[int, None]) -> Out:
            """Documentation for pred1."""
            return  # type: ignore

        @predicate
        @staticmethod
        def pred2(t1: str, t2: int, t3: Node[int, None]) -> Out:
            return  # type: ignore

        @predicate
        @staticmethod
        def pred3() -> Out:
            return  # type: ignore

        @clause(
            predicate="pred1",
            t1=Var("x"),
            t2=Var(),
            t3=Var(),
        )
        def _clause1(self, x: str) -> Out:
            return self.alg.comb2(x, self.alg.unit(), self.alg.unit())

        @clause(
            predicate="pred1",
            t1=Var(),
            t2=Var("y"),
            t3=Var(),
        )
        def _clause2(self, y: int) -> Out:
            return self.alg.comb1(y, self.alg.unit(), self.alg.unit())

        @clause(
            predicate="pred1",
            t1=chain(Item(Var("x")), Subseq(Var("y"))),
            t2=Var(),
            t3=Var("z"),
        )
        def _clause3(self, x: str, y: str, z: Node[int, None]) -> Out:
            return self.alg.comb2(x + x, self.pred2(t1=y, t2=0, t3=z), self.alg.unit())

        @clause(
            predicates={"pred1", "pred2"},
            t1=Empty(),
            t2=Zero(),
            t3=Var(),
        )
        def _clause4(self) -> Out:
            return self.alg.comb1(13, self.alg.unit(), self.alg.unit())

    gram = ExampleGrammar(
        ExampleSignature[int](
            null=lambda: 0,
            choose=operator.add,
            unit=lambda: 1,
            comb1=lambda x, y, z: 7 * x * y * z,
            comb2=lambda x, y, z: 11 * int(x) * y * z,
        )
    )

    assert gram.predicates.keys() == {"pred1", "pred2", "pred3"}
    assert gram.predicates["pred1"].name == "pred1"
    assert gram.predicates["pred1"].doc == "Documentation for pred1."
    assert gram.predicates["pred1"].parameters.keys() == {"t1", "t2", "t3"}
    assert gram.predicates["pred2"].name == "pred2"
    assert gram.predicates["pred2"].doc is None
    assert gram.predicates["pred2"].parameters.keys() == {"t1", "t2", "t3"}
    assert gram.predicates["pred3"].name == "pred3"
    assert gram.predicates["pred3"].doc is None
    assert gram.predicates["pred3"].parameters.keys() == set()

    assert gram.clauses.keys() == {"pred1", "pred2"}
    assert gram.clauses["pred1"] == [
        ExampleGrammar._clause1,
        ExampleGrammar._clause2,
        ExampleGrammar._clause3,
        ExampleGrammar._clause4,
    ]
    assert gram.clauses["pred2"] == [ExampleGrammar._clause4]

    assert gram.pred1(t1="4", t2=7, t3=Node(8)) == 11 * 4 + 7 * 7 + 11 * 44 * 13 * 7

    assert gram.memo == {
        "pred1": {
            Map(
                {
                    "t1": "4",
                    "t2": 7,
                    "t3": Node(8),
                }
            ): (11 * 4 + 7 * 7 + 11 * 44 * 13 * 7),
        },
        "pred2": {
            Map(
                {
                    "t1": "",
                    "t2": 0,
                    "t3": Node(8),
                }
            ): (13 * 7),
        },
    }
