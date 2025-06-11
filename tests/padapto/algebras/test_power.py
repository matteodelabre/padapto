import operator
from dataclasses import dataclass
from math import inf
from typing import cast

from padapto.algebras.join import join
from padapto.algebras.lex import lex
from padapto.algebras.power import power
from padapto.algebras.signature import get_algebra_metadata
from padapto.collections import Multiset

from .test_signature import SemiRing, check_semiring


def test_power_simple() -> None:
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    all_trop = cast(SemiRing[Multiset[int | float]], power(tropical))

    assert all_trop.null() == Multiset()
    assert all_trop.unit() == Multiset((0,))

    assert all_trop.choose(
        Multiset((2, 3, 13)),
        Multiset((3, 5, 17)),
    ) == Multiset((2, 3, 3, 5, 13, 17))

    assert all_trop.combine(
        Multiset((2, 3, 13)),
        Multiset((3, 5, 17)),
    ) == Multiset((2 + 3, 2 + 5, 2 + 17, 3 + 3, 3 + 5, 3 + 17, 13 + 3, 13 + 5, 13 + 17))

    sample_values: tuple[Multiset[int | float], ...] = (
        Multiset((7,)),
        Multiset((5, 7, 23)),
        Multiset((7, 13, 17)),
    )
    check_semiring(all_trop, sample_values, conservative=False)


def test_power_order() -> None:
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    all_trop = cast(SemiRing[Multiset[int | float]], power(tropical, order=True))

    assert all_trop.null() == Multiset()
    assert all_trop.unit() == Multiset((0,))

    assert list(
        all_trop.choose(
            Multiset((2, 3, 13)),
            Multiset((3, 5, 17)),
        )
    ) == [2, 3, 3, 5, 13, 17]

    assert list(
        all_trop.combine(
            Multiset((2, 3, 13)),
            Multiset((3, 5, 17)),
        )
    ) == [2 + 3, 3 + 3, 2 + 5, 3 + 5, 13 + 3, 13 + 5, 2 + 17, 3 + 17, 13 + 17]

    sample_values: tuple[Multiset[int | float], ...] = (
        Multiset((7,)),
        Multiset((5, 7, 23)),
        Multiset((7, 13, 17)),
    )
    check_semiring(all_trop, sample_values, conservative=False)


def test_power_generate() -> None:
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    generator = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))

    assert generator.null() == Multiset()
    assert generator.unit() == Multiset(((),))

    assert generator.choose(
        Multiset((("a", "b"), ("c",))),
        Multiset((("d",), ("e", "f"))),
    ) == Multiset((("a", "b"), ("c",), ("d",), ("e", "f")))

    assert generator.combine(
        Multiset((("a", "b"), ("c",))),
        Multiset((("d",), ("e", "f"))),
    ) == Multiset((("a", "b", "d"), ("a", "b", "e", "f"), ("c", "d"), ("c", "e", "f")))

    sample_values: tuple[Multiset[tuple[str, ...]], ...] = (
        Multiset((("a",),)),
        Multiset((("a", "b", "c"),)),
        Multiset((("a", "b"), ("c", "d", "e"))),
    )
    check_semiring(generator, sample_values, conservative=False)


@dataclass(frozen=True, slots=True)
class CostDistanceCount:
    cost: int | float
    distance: int | float
    count: int


def test_power_lex():
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )
    integers = SemiRing[int](
        null=lambda: 0,
        choose=operator.add,
        unit=lambda: 1,
        combine=operator.mul,
    )

    algebra = cast(
        SemiRing[Multiset[CostDistanceCount]],
        power(
            lex(
                join(
                    CostDistanceCount,
                    cost=tropical,
                    distance=tropical,
                    count=integers,
                ),
                "cost",
                "distance",
            ),
            order=True,
        ),
    )

    assert list(
        algebra.choose(
            Multiset(
                (
                    CostDistanceCount(cost=3, distance=3, count=5),
                    CostDistanceCount(cost=5, distance=2, count=3),
                    CostDistanceCount(cost=6, distance=3, count=1),
                )
            ),
            Multiset(
                (
                    CostDistanceCount(cost=3, distance=2, count=6),
                    CostDistanceCount(cost=5, distance=2, count=4),
                    CostDistanceCount(cost=7, distance=3, count=2),
                )
            ),
        )
    ) == [
        CostDistanceCount(cost=3, distance=2, count=6),
        CostDistanceCount(cost=3, distance=3, count=5),
        CostDistanceCount(cost=5, distance=2, count=3),
        CostDistanceCount(cost=5, distance=2, count=4),
        CostDistanceCount(cost=6, distance=3, count=1),
        CostDistanceCount(cost=7, distance=3, count=2),
    ]

    check_semiring(
        algebra,
        (
            Multiset((CostDistanceCount(cost=1, distance=1, count=1),)),
            Multiset((CostDistanceCount(cost=3, distance=2, count=7),)),
            Multiset(
                (
                    CostDistanceCount(cost=3, distance=2, count=5),
                    CostDistanceCount(cost=5, distance=3, count=3),
                    CostDistanceCount(cost=7, distance=2, count=1),
                )
            ),
        ),
        conservative=False,
    )


def test_power_power() -> None:
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    double = cast(SemiRing[Multiset[Multiset[int | float]]], power(power(tropical)))

    assert double.null() == Multiset()
    assert double.unit() == Multiset((Multiset((0,)),))

    assert double.choose(
        Multiset((Multiset((2, 3)), Multiset((13,)))),
        Multiset((Multiset((3,)), Multiset((5, 17)))),
    ) == Multiset(
        (
            Multiset((2, 3)),
            Multiset((13,)),
            Multiset((3,)),
            Multiset((5, 17)),
        )
    )

    assert double.combine(
        Multiset((Multiset((2, 3)), Multiset((13,)))),
        Multiset((Multiset((3,)), Multiset((5, 17)))),
    ) == Multiset(
        (
            Multiset((2 + 3, 3 + 3)),
            Multiset((2 + 5, 2 + 17, 3 + 5, 3 + 17)),
            Multiset((13 + 3,)),
            Multiset((13 + 5, 13 + 17)),
        )
    )

    sample_values: tuple[Multiset[Multiset[int | float]], ...] = (
        Multiset((Multiset((7,)),)),
        Multiset((Multiset((5, 7)), Multiset((23,)))),
        Multiset((Multiset((7,)), Multiset((13, 17)))),
    )
    check_semiring(double, sample_values, conservative=False)


def test_power_metadata():
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    generator = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))
    assert get_algebra_metadata(generator, power) == free
