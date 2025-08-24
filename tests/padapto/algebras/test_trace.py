from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from sowing import Node

from padapto.algebras.signature import Signature
from padapto.algebras.trace import (
    Circuit,
    CircuitData,
    enumerate_solutions,
    get_solution,
    trace,
)

from .test_signature import SemiRing


@dataclass(frozen=True)
class MultiSemiRing[T](Signature[T]):
    unit1: Callable[[], T]
    unit2: Callable[[], T]
    unit3: Callable[[], T]
    combine: Callable[[T, T], T]


def test_trace_product():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert alg.null() == Node(CircuitData(operator="null"))
    assert alg.unit1() == Node(CircuitData(operator="unit1"))

    assert alg.combine(alg.unit1(), alg.unit1()) == (
        Node(CircuitData(operator="combine"))
        .add(Node(CircuitData(operator="unit1")))
        .add(Node(CircuitData(operator="unit1")))
    )

    assert alg.combine(alg.unit1(), alg.null()) == alg.null()
    assert alg.combine(alg.null(), alg.unit1()) == alg.null()

    assert alg.combine(alg.unit1(), alg.combine(alg.unit1(), alg.unit1())) == (
        Node(CircuitData(operator="combine"))
        .add(Node(CircuitData(operator="unit1")))
        .add(
            Node(CircuitData(operator="combine"))
            .add(Node(CircuitData(operator="unit1")))
            .add(Node(CircuitData(operator="unit1")))
        )
    )


def test_trace_choice():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert alg.choose(alg.unit1(), alg.unit2()) == (
        Node(CircuitData(operator="choose")).add(alg.unit1()).add(alg.unit2())
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


def test_trace_enumerate_one():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert (
        get_solution(
            alg.choose(
                alg.unit1(),
                alg.unit2(),
            )
        )
        == alg.unit1()
    )


def test_trace_enumerate_all():
    alg = cast(SemiRing[Circuit], trace(MultiSemiRing))

    assert list(
        enumerate_solutions(
            alg.choose(
                alg.unit1(),
                alg.unit2(),
            )
        )
    ) == [alg.unit1(), alg.unit2()]

    assert list(
        enumerate_solutions(
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
