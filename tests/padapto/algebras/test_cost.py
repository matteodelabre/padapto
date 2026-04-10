from collections.abc import Callable
from dataclasses import dataclass
from math import exp, inf

from padapto.algebras.cost import add_optimizer, boltzmann
from padapto.algebras.signature import Signature


@dataclass(frozen=True)
class OutSemiRing[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[str, T, str, T], T]


def test_cost_min() -> None:
    min_cost: OutSemiRing[float] = add_optimizer(
        OutSemiRing,
        choose=min,
        combine=lambda x, y: x != y,
    )

    assert min_cost.null() == inf
    assert min_cost.choose(3, 7) == 3
    assert min_cost.multichoose(3, 7, 1, 4) == 1
    assert min_cost.unit() == 0
    assert min_cost.combine("a", 3, "b", 7) == 11
    assert min_cost.combine("a", 3, "a", 7) == 10


def test_cost_max() -> None:
    max_value: OutSemiRing[float] = add_optimizer(
        OutSemiRing,
        choose=max,
        combine=lambda x, y: x == y,
    )

    assert max_value.null() == -inf
    assert max_value.choose(3, 7) == 7
    assert max_value.multichoose(3, 7, 1, 4) == 7
    assert max_value.unit() == 0
    assert max_value.combine("a", 3, "b", 7) == 10
    assert max_value.combine("a", 3, "a", 7) == 11


def test_cost_boltzmann() -> None:
    boltz: OutSemiRing[float] = boltzmann(
        OutSemiRing,
        temperature=2,
        combine=lambda x, y: x != y,
    )

    assert boltz.null() == 0
    assert boltz.choose(3, 7) == 10
    assert boltz.multichoose(3, 7, 1, 4) == 15
    assert boltz.unit() == 1
    assert boltz.combine("a", exp(-3 / 2), "b", exp(-7 / 2)) == exp(-11 / 2)
    assert boltz.combine("a", exp(-3 / 2), "a", exp(-7 / 2)) == exp(-10 / 2)
