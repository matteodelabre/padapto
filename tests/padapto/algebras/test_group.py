import operator
from dataclasses import dataclass
from math import inf
from typing import cast

import pytest

from padapto.algebras.group import group
from padapto.algebras.join import join
from padapto.algebras.power import power
from padapto.collections import Multiset

from .test_signature import SemiRing, check_semiring


@dataclass(frozen=True, slots=True)
class CostCount:
    cost: int | float
    count: int


def test_group_count():
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
        SemiRing[Multiset[CostCount]],
        group(power(join(CostCount, cost=tropical, count=integers)), "cost"),
    )

    assert algebra.null() == Multiset()
    assert algebra.unit() == Multiset((CostCount(cost=0, count=1),))

    assert algebra.choose(
        Multiset(
            (
                CostCount(cost=2, count=3),
                CostCount(cost=3, count=5),
            )
        ),
        Multiset(
            (
                CostCount(cost=3, count=7),
                CostCount(cost=5, count=11),
            )
        ),
    ) == Multiset(
        (
            CostCount(cost=2, count=3),
            CostCount(cost=3, count=5 + 7),
            CostCount(cost=5, count=11),
        )
    )

    assert algebra.combine(
        Multiset(
            (
                CostCount(cost=2, count=3),
                CostCount(cost=3, count=5),
            )
        ),
        Multiset(
            (
                CostCount(cost=3, count=7),
                CostCount(cost=4, count=11),
            )
        ),
    ) == Multiset(
        (
            CostCount(cost=5, count=3 * 7),
            CostCount(cost=6, count=3 * 11 + 5 * 7),
            CostCount(cost=7, count=5 * 11),
        )
    )

    check_semiring(
        algebra,
        (
            Multiset((CostCount(cost=3, count=1),)),
            Multiset((CostCount(cost=2, count=3), CostCount(cost=3, count=5))),
            Multiset((CostCount(cost=1, count=2), CostCount(cost=2, count=4))),
        ),
        conservative=False,
    )


def test_group_twice():
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
        SemiRing[Multiset[CostCount]],
        group(
            group(power(join(CostCount, cost=tropical, count=integers)), "cost"),
            "cost",
        ),
    )

    assert algebra.null() == Multiset()
    assert algebra.unit() == Multiset((CostCount(cost=0, count=1),))

    assert algebra.choose(
        Multiset(
            (
                CostCount(cost=2, count=3),
                CostCount(cost=3, count=5),
            )
        ),
        Multiset(
            (
                CostCount(cost=3, count=7),
                CostCount(cost=5, count=11),
            )
        ),
    ) == Multiset(
        (
            CostCount(cost=2, count=3),
            CostCount(cost=3, count=5 + 7),
            CostCount(cost=5, count=11),
        )
    )


@dataclass(frozen=True, slots=True)
class CostSolutions:
    cost: int | float
    solutions: Multiset[tuple[str, ...]]


def test_group_sets():
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    generator = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))

    algebra = cast(
        SemiRing[Multiset[CostSolutions]],
        group(power(join(CostSolutions, cost=tropical, solutions=generator)), "cost"),
    )

    assert algebra.null() == Multiset()
    assert algebra.unit() == Multiset(
        (CostSolutions(cost=0, solutions=Multiset(((),))),)
    )

    assert algebra.choose(
        Multiset(
            (
                CostSolutions(cost=2, solutions=Multiset((("a", "b"), ("c", "d")))),
                CostSolutions(cost=3, solutions=Multiset((("a", "b", "c"),))),
            )
        ),
        Multiset(
            (
                CostSolutions(cost=2, solutions=Multiset((("b", "c"),))),
                CostSolutions(cost=3, solutions=Multiset((("a", "b", "c"), ("c",)))),
                CostSolutions(cost=4, solutions=Multiset((("a", "b", "c", "d"),))),
            )
        ),
    ) == Multiset(
        (
            CostSolutions(
                cost=2,
                solutions=Multiset((("a", "b"), ("b", "c"), ("c", "d"))),
            ),
            CostSolutions(
                cost=3,
                solutions=Multiset((("a", "b", "c"), ("a", "b", "c"), ("c",))),
            ),
            CostSolutions(cost=4, solutions=Multiset((("a", "b", "c", "d"),))),
        )
    )

    check_semiring(
        algebra,
        (
            Multiset((CostSolutions(cost=1, solutions=Multiset((("a",),))),)),
            Multiset(
                (
                    CostSolutions(cost=2, solutions=Multiset((("a", "b"), ("c", "d")))),
                    CostSolutions(cost=3, solutions=Multiset((("a", "b", "c"),))),
                )
            ),
            Multiset(
                (
                    CostSolutions(cost=2, solutions=Multiset((("b", "c"),))),
                    CostSolutions(
                        cost=3, solutions=Multiset((("a", "b", "c"), ("c",)))
                    ),
                    CostSolutions(cost=4, solutions=Multiset((("a", "b", "c", "d"),))),
                )
            ),
        ),
        conservative=False,
    )


def test_group_none():
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
        SemiRing[Multiset[CostCount]],
        group(power(join(CostCount, cost=tropical, count=integers))),
    )

    assert algebra.choose(
        Multiset((CostCount(cost=2, count=3),)),
        Multiset((CostCount(cost=3, count=7),)),
    ) == Multiset((CostCount(cost=2, count=3 + 7),))

    assert algebra.combine(
        Multiset((CostCount(cost=2, count=3),)),
        Multiset((CostCount(cost=3, count=7),)),
    ) == Multiset((CostCount(cost=2 + 3, count=3 * 7),))

    check_semiring(
        algebra,
        (
            Multiset((CostCount(cost=3, count=1),)),
            Multiset((CostCount(cost=2, count=3),)),
            Multiset((CostCount(cost=1, count=2),)),
        ),
        conservative=False,
    )


def test_group_invalid():
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    with pytest.raises(
        TypeError,
        match="group: provided algebra is not a power algebra",
    ):
        group(tropical, "a", "b", "c")

    pow_tropical = power(tropical)

    with pytest.raises(
        TypeError,
        match="group: provided algebra is not a joined algebra",
    ):
        group(pow_tropical, "a", "b", "c")

    joined = power(join(a=tropical, b=tropical))

    with pytest.raises(TypeError, match="group: 'c' is not a subalgebra field"):
        group(joined, "a", "b", "c")
