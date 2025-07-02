from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from padapto.algebras.signature import Signature
from padapto.algebras.trace import CandidateNode, enumerate_candidates, trace

from .test_signature import SemiRing


@dataclass(frozen=True)
class MultiSemiRing[T](Signature[T]):
    unit1: Callable[[], T]
    unit2: Callable[[], T]
    unit3: Callable[[], T]
    combine: Callable[[T, T], T]


def test_trace_product():
    alg = cast(SemiRing[CandidateNode], trace(MultiSemiRing))

    assert alg.null() == CandidateNode(("null",))
    assert alg.unit1() == CandidateNode(("unit1",))

    assert alg.combine(alg.unit1(), alg.unit1()) == (
        CandidateNode(("combine",))
        .add(CandidateNode(("unit1",)))
        .add(CandidateNode(("unit1",)))
    )

    assert alg.combine(alg.unit1(), alg.null()) == alg.null()
    assert alg.combine(alg.null(), alg.unit1()) == alg.null()

    assert alg.combine(alg.unit1(), alg.combine(alg.unit1(), alg.unit1())) == (
        CandidateNode(("combine",))
        .add(CandidateNode(("unit1",)))
        .add(
            CandidateNode(("combine",))
            .add(CandidateNode(("unit1",)))
            .add(CandidateNode(("unit1",)))
        )
    )


def test_trace_choice_single():
    alg = cast(SemiRing[CandidateNode], trace(MultiSemiRing, single=True))

    assert alg.choose(alg.unit1(), alg.unit2()) == alg.unit1()
    assert alg.choose(alg.unit1(), alg.null()) == alg.unit1()
    assert alg.choose(alg.null(), alg.unit2()) == alg.unit2()
    assert alg.choose(alg.null(), alg.null()) == alg.null()
    assert alg.multichoose() == alg.null()


def test_trace_choice_all():
    alg = cast(SemiRing[CandidateNode], trace(MultiSemiRing, single=False))

    assert alg.choose(alg.unit1(), alg.unit2()) == (
        CandidateNode(("choose",)).add(alg.unit1()).add(alg.unit2())
    )

    # Neutral element
    assert alg.choose(alg.null(), alg.null()) == alg.null()
    assert alg.multichoose() == alg.null()
    assert alg.choose(alg.unit1(), alg.null()) == alg.unit1()
    assert alg.choose(alg.null(), alg.unit1()) == alg.unit1()

    # Commutativity
    assert alg.choose(alg.unit1(), alg.unit2()) == alg.choose(alg.unit2(), alg.unit1())

    # Idempotency
    assert alg.choose(alg.unit1(), alg.unit1()) == alg.unit1()

    # Associativity
    assert alg.choose(alg.unit1(), alg.choose(alg.unit2(), alg.unit3())) == alg.choose(
        alg.choose(alg.unit1(), alg.unit2()), alg.unit3()
    )

    # Idempotency with multiple arguments
    assert alg.choose(alg.unit1(), alg.choose(alg.unit2(), alg.unit1())) == alg.choose(
        alg.unit1(), alg.unit2()
    )


def test_trace_enumerate_candidates():
    alg = cast(SemiRing[CandidateNode], trace(MultiSemiRing, single=False))

    assert list(
        enumerate_candidates(
            alg.choose(
                alg.unit1(),
                alg.unit2(),
            )
        )
    ) == [alg.unit1(), alg.unit2()]

    assert list(
        enumerate_candidates(
            alg.combine(
                alg.choose(
                    alg.unit1(),
                    alg.unit2(),
                ),
                alg.choose(
                    alg.unit2(),
                    alg.unit3(),
                ),
            )
        )
    ) == [
        alg.combine(alg.unit1(), alg.unit2()),
        alg.combine(alg.unit1(), alg.unit3()),
        alg.combine(alg.unit2(), alg.unit2()),
        alg.combine(alg.unit2(), alg.unit3()),
    ]
