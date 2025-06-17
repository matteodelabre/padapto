import operator
from dataclasses import dataclass
from math import inf

from padapto.algebras.join import join
from padapto.algebras.pareto import pareto
from padapto.algebras.power import power
from padapto.algebras.signature import get_algebra_parent
from padapto.collections import Multiset

from .test_signature import SemiRing, check_semiring


@dataclass(frozen=True, slots=True)
class TwoFields:
    first: int | float
    second: int | float


def test_pareto_two():
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    algebra = (
        join(TwoFields, first=tropical, second=tropical)
        | power()
        | pareto("first", "second")
    )

    assert algebra.choose(
        Multiset((TwoFields(first=1, second=2),)),
        Multiset((TwoFields(first=2, second=1),)),
    ) == Multiset(
        (
            TwoFields(first=1, second=2),
            TwoFields(first=2, second=1),
        )
    )

    assert algebra.combine(
        Multiset((TwoFields(first=1, second=2),)),
        Multiset((TwoFields(first=2, second=1),)),
    ) == Multiset((TwoFields(first=3, second=3),))

    assert algebra.choose(
        Multiset(
            (
                TwoFields(first=3, second=5),
                TwoFields(first=4, second=3),
            )
        ),
        Multiset(
            (
                TwoFields(first=5, second=3),
                TwoFields(first=3, second=4),
            )
        ),
    ) == Multiset(
        (
            TwoFields(first=4, second=3),
            TwoFields(first=3, second=4),
        )
    )

    assert algebra.combine(
        Multiset((TwoFields(first=3, second=5), TwoFields(first=4, second=3))),
        Multiset((TwoFields(first=5, second=3), TwoFields(first=3, second=4))),
    ) == Multiset(
        (
            TwoFields(first=7, second=7),
            TwoFields(first=6, second=9),
            TwoFields(first=9, second=6),
        )
    )

    check_semiring(
        algebra,
        (
            Multiset((TwoFields(first=2, second=3),)),
            Multiset((TwoFields(first=1, second=2),)),
            Multiset((TwoFields(first=3, second=2),)),
            Multiset((TwoFields(first=3, second=2), TwoFields(first=1, second=5))),
        ),
        conservative=False,
    )


@dataclass(frozen=True, slots=True)
class TwoFieldsData:
    first: int | float
    second: int | float
    data: int


def test_pareto_partial():
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

    algebra = (
        join(TwoFieldsData, first=tropical, second=tropical, data=integers)
        | power()
        | pareto("first", "second")
    )

    assert algebra.choose(
        Multiset((TwoFieldsData(first=1, second=2, data=2),)),
        Multiset((TwoFieldsData(first=2, second=1, data=3),)),
    ) == Multiset(
        (
            TwoFieldsData(first=1, second=2, data=2),
            TwoFieldsData(first=2, second=1, data=3),
        )
    )

    assert algebra.choose(
        Multiset((TwoFieldsData(first=2, second=2, data=2),)),
        Multiset((TwoFieldsData(first=2, second=1, data=3),)),
    ) == Multiset((TwoFieldsData(first=2, second=1, data=3),))

    assert algebra.choose(
        Multiset((TwoFieldsData(first=2, second=2, data=2),)),
        Multiset((TwoFieldsData(first=2, second=2, data=3),)),
    ) == Multiset((TwoFieldsData(first=2, second=2, data=2 + 3),))

    assert algebra.combine(
        Multiset((TwoFieldsData(first=1, second=2, data=2),)),
        Multiset((TwoFieldsData(first=2, second=1, data=3),)),
    ) == Multiset((TwoFieldsData(first=3, second=3, data=2 * 3),))

    assert algebra.combine(
        Multiset(
            (
                TwoFieldsData(first=3, second=2, data=7),
                TwoFieldsData(first=1, second=5, data=11),
            )
        ),
        Multiset(
            (
                TwoFieldsData(first=3, second=2, data=7),
                TwoFieldsData(first=1, second=5, data=11),
            )
        ),
    ) == Multiset(
        (
            TwoFieldsData(first=6, second=4, data=7 * 7),
            TwoFieldsData(first=4, second=7, data=7 * 11 + 7 * 11),
            TwoFieldsData(first=2, second=10, data=11 * 11),
        )
    )

    assert algebra.combine(
        Multiset(
            (
                TwoFieldsData(first=4, second=7, data=2),
                TwoFieldsData(first=6, second=5, data=7),
            )
        ),
        Multiset(
            (
                TwoFieldsData(first=6, second=5, data=5),
                TwoFieldsData(first=5, second=6, data=11),
                TwoFieldsData(first=7, second=4, data=13),
            )
        ),
    ) == Multiset(
        (
            TwoFieldsData(first=9, second=13, data=2 * 11),
            TwoFieldsData(first=10, second=12, data=2 * 5),
            TwoFieldsData(first=11, second=11, data=7 * 11 + 2 * 13),
            TwoFieldsData(first=12, second=10, data=5 * 7),
            TwoFieldsData(first=13, second=9, data=7 * 13),
        )
    )

    check_semiring(
        algebra,
        (
            Multiset((TwoFieldsData(first=2, second=3, data=2),)),
            Multiset((TwoFieldsData(first=1, second=2, data=3),)),
            Multiset((TwoFieldsData(first=3, second=2, data=5),)),
            Multiset(
                (
                    TwoFieldsData(first=3, second=2, data=7),
                    TwoFieldsData(first=1, second=5, data=11),
                )
            ),
        ),
        conservative=False,
    )


def test_pareto_none():
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

    algebra = (
        join(TwoFieldsData, first=tropical, second=tropical, data=integers)
        | power()
        | pareto()
    )

    assert algebra.choose(
        Multiset((TwoFieldsData(first=1, second=2, data=2),)),
        Multiset((TwoFieldsData(first=2, second=1, data=3),)),
    ) == Multiset((TwoFieldsData(first=1, second=1, data=2 + 3),))

    check_semiring(
        algebra,
        (
            Multiset((TwoFieldsData(first=2, second=3, data=2),)),
            Multiset((TwoFieldsData(first=1, second=2, data=3),)),
            Multiset((TwoFieldsData(first=3, second=2, data=5),)),
        ),
        conservative=False,
    )


def test_pareto_metadata():
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

    joined = join(TwoFieldsData, first=tropical, second=tropical, data=integers)
    powerset = joined | power()
    algebra = powerset | pareto("first", "second")
    assert get_algebra_parent(algebra) == get_algebra_parent(powerset)
