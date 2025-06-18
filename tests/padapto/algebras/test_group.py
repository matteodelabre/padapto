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
        join(CostCount, cost=tropical, count=integers) | power() | group("cost"),
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
        (
            join(CostCount, cost=tropical, count=integers)
            | power()
            | group("cost")
            | group("cost")
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
class NestedCostCount:
    left: CostCount
    right: int


def test_group_nested():
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
        SemiRing[NestedCostCount],
        join(
            NestedCostCount,
            left=join(CostCount, cost=tropical, count=integers),
            right=integers,
        )
        | power()
        | group("left.cost"),
    )

    assert algebra.null() == Multiset()
    assert algebra.unit() == Multiset(
        (NestedCostCount(left=CostCount(cost=0, count=1), right=1),)
    )

    assert algebra.choose(
        Multiset(
            (
                NestedCostCount(left=CostCount(cost=2, count=3), right=7),
                NestedCostCount(left=CostCount(cost=3, count=5), right=11),
            )
        ),
        Multiset(
            (
                NestedCostCount(left=CostCount(cost=3, count=7), right=11),
                NestedCostCount(left=CostCount(cost=5, count=11), right=13),
            )
        ),
    ) == Multiset(
        (
            NestedCostCount(left=CostCount(cost=2, count=3), right=7),
            NestedCostCount(left=CostCount(cost=3, count=5 + 7), right=11 + 11),
            NestedCostCount(left=CostCount(cost=5, count=11), right=13),
        )
    )

    assert algebra.combine(
        Multiset(
            (
                NestedCostCount(left=CostCount(cost=2, count=3), right=7),
                NestedCostCount(left=CostCount(cost=3, count=5), right=11),
            )
        ),
        Multiset(
            (
                NestedCostCount(left=CostCount(cost=3, count=7), right=11),
                NestedCostCount(left=CostCount(cost=4, count=11), right=13),
            )
        ),
    ) == Multiset(
        (
            NestedCostCount(left=CostCount(cost=5, count=3 * 7), right=7 * 11),
            NestedCostCount(
                left=CostCount(cost=6, count=3 * 11 + 5 * 7),
                right=7 * 13 + 11 * 11,
            ),
            NestedCostCount(left=CostCount(cost=7, count=5 * 11), right=11 * 13),
        )
    )

    check_semiring(
        algebra,
        (
            Multiset((NestedCostCount(left=CostCount(cost=3, count=1), right=2),)),
            Multiset(
                (
                    NestedCostCount(left=CostCount(cost=2, count=3), right=5),
                    NestedCostCount(left=CostCount(cost=3, count=5), right=7),
                )
            ),
            Multiset(
                (
                    NestedCostCount(left=CostCount(cost=1, count=2), right=7),
                    NestedCostCount(left=CostCount(cost=2, count=4), right=2),
                )
            ),
        ),
        conservative=False,
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

    generator = cast(SemiRing[Multiset[tuple[str, ...]]], free | power())

    algebra = cast(
        SemiRing[Multiset[CostSolutions]],
        (
            join(CostSolutions, cost=tropical, solutions=generator)
            | power()
            | group("cost")
        ),
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
        join(CostCount, cost=tropical, count=integers) | power() | group(),
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
        tropical | group("a", "b", "c")

    pow_tropical = tropical | power()

    with pytest.raises(
        TypeError,
        match="provided algebra is not joined",
    ):
        pow_tropical | group("a", "b", "c")

    joined = join(a=tropical, b=tropical) | power()

    with pytest.raises(
        AttributeError, match="'c' is not a field of the provided algebra"
    ):
        joined | group("a", "b", "c")
