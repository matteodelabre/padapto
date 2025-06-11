import operator
from dataclasses import dataclass
from math import inf
from typing import cast

import pytest

from padapto.algebras.join import join
from padapto.algebras.lex import lex
from padapto.algebras.signature import get_algebra_metadata

from .test_signature import SemiRing, check_semiring


@dataclass(frozen=True, slots=True)
class CostCount:
    cost: int | float
    count: int


def test_lex_single():
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

    joined = cast(
        SemiRing[CostCount],
        join(CostCount, cost=tropical, count=integers),
    )

    select = lex(joined, "cost")

    assert select.choose(
        CostCount(cost=2, count=10),
        CostCount(cost=3, count=7),
    ) == CostCount(cost=2, count=10)

    assert select.choose(
        CostCount(cost=3, count=10),
        CostCount(cost=2, count=7),
    ) == CostCount(cost=2, count=7)

    assert select.choose(
        CostCount(cost=2, count=10),
        CostCount(cost=2, count=7),
    ) == CostCount(cost=2, count=7 + 10)

    assert select.combine(
        CostCount(cost=2, count=10),
        CostCount(cost=3, count=7),
    ) == CostCount(cost=2 + 3, count=7 * 10)

    check_semiring(
        select,
        (
            CostCount(cost=2, count=10),
            CostCount(cost=3, count=7),
        ),
        conservative=False,
    )


@dataclass(frozen=True, slots=True)
class CostDistanceCount:
    cost: int | float
    distance: int | float
    count: int


def test_lex_multiple():
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

    joined = cast(
        SemiRing[CostDistanceCount],
        join(CostDistanceCount, cost=tropical, distance=tropical, count=integers),
    )

    sel_cost_distance = lex(joined, "cost", "distance")
    sel_distance_cost = lex(joined, "distance", "cost")

    assert sel_cost_distance.choose(
        CostDistanceCount(cost=2, distance=7, count=10),
        CostDistanceCount(cost=3, distance=5, count=7),
    ) == CostDistanceCount(cost=2, distance=7, count=10)

    assert sel_distance_cost.choose(
        CostDistanceCount(cost=2, distance=7, count=10),
        CostDistanceCount(cost=3, distance=5, count=7),
    ) == CostDistanceCount(cost=3, distance=5, count=7)

    assert sel_cost_distance.choose(
        CostDistanceCount(cost=2, distance=7, count=10),
        CostDistanceCount(cost=2, distance=5, count=7),
    ) == CostDistanceCount(cost=2, distance=5, count=7)

    assert sel_distance_cost.choose(
        CostDistanceCount(cost=2, distance=5, count=10),
        CostDistanceCount(cost=3, distance=5, count=7),
    ) == CostDistanceCount(cost=2, distance=5, count=10)

    assert sel_cost_distance.choose(
        CostDistanceCount(cost=2, distance=5, count=10),
        CostDistanceCount(cost=2, distance=5, count=7),
    ) == CostDistanceCount(cost=2, distance=5, count=7 + 10)

    check_semiring(
        sel_cost_distance,
        (
            CostDistanceCount(cost=2, distance=7, count=10),
            CostDistanceCount(cost=3, distance=5, count=7),
            CostDistanceCount(cost=4, distance=4, count=3),
            CostDistanceCount(cost=1, distance=2, count=5),
        ),
        conservative=False,
    )


def test_lex_none():
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

    joined = cast(
        SemiRing[CostDistanceCount],
        join(CostDistanceCount, cost=tropical, distance=tropical, count=integers),
    )

    lexed_none = lex(joined)

    assert lexed_none.choose(
        CostDistanceCount(cost=2, distance=7, count=10),
        CostDistanceCount(cost=3, distance=5, count=7),
    ) == CostDistanceCount(cost=2, distance=5, count=17)

    check_semiring(
        lexed_none,
        (
            CostDistanceCount(cost=2, distance=7, count=10),
            CostDistanceCount(cost=3, distance=5, count=7),
            CostDistanceCount(cost=4, distance=4, count=3),
            CostDistanceCount(cost=1, distance=2, count=5),
        ),
        conservative=False,
    )


def test_lex_invalid():
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

    with pytest.raises(
        TypeError, match="lex: provided algebra is not a joined algebra"
    ):
        lex(tropical, "unknown")

    joined = cast(
        SemiRing[CostCount],
        join(CostCount, cost=tropical, count=integers),
    )

    with pytest.raises(TypeError, match="lex: 'unknown' is not a subalgebra field"):
        lex(joined, "unknown")

    revselect = lex(joined, "count")

    with pytest.raises(TypeError, match="lex: order for field 'count' is not total"):
        assert revselect.choose(
            CostCount(cost=2, count=10),
            CostCount(cost=3, count=7),
        ) == CostCount(cost=2, count=10)


def test_lex_metadata():
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

    joined = cast(
        SemiRing[CostCount],
        join(CostCount, cost=tropical, count=integers),
    )
    subalgebras = get_algebra_metadata(joined, join)

    select_cost = lex(joined, "cost")
    assert get_algebra_metadata(select_cost, lex) == (joined, ("cost",))
    assert get_algebra_metadata(select_cost, join) == subalgebras

    select_cost_count = lex(select_cost, "count")
    assert get_algebra_metadata(select_cost_count, lex) == (joined, ("count", "cost"))
    assert get_algebra_metadata(select_cost_count, join) == subalgebras

    select_cost_count2 = lex(joined, "count", "cost")
    assert get_algebra_metadata(select_cost_count2, lex) == (joined, ("count", "cost"))
    assert get_algebra_metadata(select_cost_count2, join) == subalgebras
