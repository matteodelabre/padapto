from collections.abc import Callable
from dataclasses import dataclass
from math import exp, inf
from typing import cast

from padapto.algebras.cost import add_optimizer, boltzmann
from padapto.algebras.signature import Signature


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
        add_optimizer(OutSemiRing, choose="min", combine=combine_cost),
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
        add_optimizer(
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

    assert boltz.null() == 0
    assert boltz.choose(3, 7) == 10
    assert boltz.multichoose(3, 7, 1, 4) == 15
    assert boltz.unit() == 1
    assert boltz.combine("a", exp(-3 / 2), "b", exp(-7 / 2)) == exp(-11 / 2)
    assert boltz.combine("a", exp(-3 / 2), "a", exp(-7 / 2)) == exp(-10 / 2)
