from collections.abc import Callable
from dataclasses import dataclass
from math import exp, inf, log
from typing import cast

from pytest import approx

from padapto.evaluation.cost import additive, boltzmann
from padapto.signature import Signature


@dataclass(frozen=True)
class OutSemiRing[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[str, T, str, T], T]


def combine_cost(x: str, y: str) -> float:
    return float(x != y)


def combine_value(x: str, y: str) -> float:
    return float(x == y)


def test_cost_min() -> None:
    min_cost = cast(
        OutSemiRing[float],
        additive(OutSemiRing, choose="min", combine=combine_cost),
    )

    assert min_cost.null() == inf
    assert min_cost.choose(3, 7) == 3
    assert min_cost.multichoose(3, 7, 1, 4) == 1
    assert min_cost.unit() == 0
    assert min_cost.combine("a", 3, "b", 7) == 11
    assert min_cost.combine("a", 3, "a", 7) == 10


def test_cost_max() -> None:
    max_value = cast(
        OutSemiRing[float],
        additive(
            OutSemiRing,
            choose="max",
            combine=combine_value,
        ),
    )

    assert max_value.null() == -inf
    assert max_value.choose(3, 7) == 7
    assert max_value.multichoose(3, 7, 1, 4) == 7
    assert max_value.unit() == 0
    assert max_value.combine("a", 3, "b", 7) == 10
    assert max_value.combine("a", 3, "a", 7) == 11


def test_cost_boltzmann() -> None:
    boltz = cast(
        OutSemiRing[float],
        boltzmann(
            OutSemiRing,
            temperature=2,
            combine=combine_cost,
        ),
    )

    assert boltz.null() == float("-inf")
    assert exp(boltz.choose(log(3), log(7))) == approx(10)
    assert exp(boltz.multichoose(log(3), log(7), log(1), log(4))) == approx(15)
    assert boltz.choose(-200_000, -300_000) == approx(-200_000)
    assert boltz.unit() == 0
    assert boltz.combine("a", -3 / 2, "b", -7 / 2) == -11 / 2
    assert boltz.combine("a", -3 / 2, "a", -7 / 2) == -10 / 2
