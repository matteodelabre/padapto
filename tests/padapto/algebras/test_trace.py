from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from padapto.algebras.signature import Signature
from padapto.algebras.trace import trace
from padapto.circuit import Circuit, make_node

from .test_signature import SemiRing


@dataclass(frozen=True)
class MultiSemiRing[T](Signature[T]):
    unit1: Callable[[], T]
    unit2: Callable[[], T]
    unit3: Callable[[], T]
    combine: Callable[[T, T], T]


def test_trace_product():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert alg.null() == make_node(operator="null")
    assert alg.unit1() == make_node(operator="unit1")

    assert alg.combine(alg.unit1(), alg.unit1()) == (
        make_node(operator="combine")
        .add(make_node(operator="unit1"))
        .add(make_node(operator="unit1"))
    )

    assert alg.combine(alg.unit1(), alg.null()) == alg.null()
    assert alg.combine(alg.null(), alg.unit1()) == alg.null()

    assert alg.combine(alg.unit1(), alg.combine(alg.unit1(), alg.unit1())) == (
        make_node(operator="combine")
        .add(make_node(operator="unit1"))
        .add(
            make_node(operator="combine")
            .add(make_node(operator="unit1"))
            .add(make_node(operator="unit1"))
        )
    )


def test_trace_choice():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert alg.choose(alg.unit1(), alg.unit2()) == (
        make_node(operator="choose").add(alg.unit1()).add(alg.unit2())
    )

    # Neutral element
    assert alg.choose(alg.null(), alg.null()) == alg.null()
    assert alg.multichoose() == alg.null()
    assert alg.choose(alg.unit1(), alg.null()) == alg.unit1()
    assert alg.choose(alg.null(), alg.unit1()) == alg.unit1()

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
