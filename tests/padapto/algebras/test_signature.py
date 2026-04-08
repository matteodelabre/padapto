import itertools
import operator
from collections.abc import Callable
from dataclasses import dataclass
from math import inf

from padapto.algebras.signature import Signature


@dataclass(frozen=True)
class SemiRing[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[T, T], T]


def check_semiring[T](
    algebra: SemiRing[T],
    values: tuple[T, ...],
    conservative: bool,
) -> None:
    values = (algebra.null(), algebra.unit(), *values)

    for value in values:
        # Check neutral elements
        assert algebra.choose(algebra.null(), value) == value
        assert algebra.choose(value, algebra.null()) == value

        assert algebra.combine(algebra.unit(), value) == value
        assert algebra.combine(value, algebra.unit()) == value

        # Check idempotency
        if conservative:
            assert algebra.choose(value, value) == value

    # Check nullary multichoose
    assert algebra.multichoose() == algebra.null()

    for value1, value2 in itertools.combinations(values, r=2):
        # Check binary multichoose
        assert algebra.multichoose(value1, value2) == algebra.choose(value1, value2)

        # Check commutativity
        assert algebra.choose(value1, value2) == algebra.choose(value2, value1)

        # Check conservativity
        if conservative:
            assert algebra.choose(value1, value2) in (value1, value2)

    for value1, value2, value3 in itertools.product(values, repeat=3):
        # Check associativity and ternary multichoose
        assert algebra.multichoose(value1, value2, value3) == algebra.choose(
            algebra.choose(value1, value2), value3
        )

        assert algebra.multichoose(value1, value2, value3) == algebra.choose(
            value1, algebra.choose(value2, value3)
        )

        # Check distributivity
        assert algebra.choose(
            algebra.combine(value1, value2),
            algebra.combine(value1, value3),
        ) == algebra.combine(value1, algebra.choose(value2, value3))


def test_signature_semirings() -> None:
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )
    check_semiring(tropical, (2, 3, 5), conservative=True)

    integers = SemiRing[int](
        null=lambda: 0,
        choose=operator.add,
        unit=lambda: 1,
        combine=operator.mul,
    )
    check_semiring(integers, (2, 3, 5), conservative=False)

    sets = SemiRing[frozenset[int]](
        null=lambda: frozenset(),
        choose=operator.or_,
        unit=lambda: frozenset({0}),
        combine=lambda x, y: frozenset({a + b for a in x for b in y}),
    )
    check_semiring(sets, (frozenset({2, 5}), frozenset({3, 11})), conservative=False)


def test_natural_order() -> None:
    tropical = SemiRing[int | float](
        null=lambda: inf,
        choose=min,
        unit=lambda: 0,
        combine=operator.add,
    )

    tropical_le = tropical.natural_order()
    assert tropical_le(0, inf)
    assert not tropical_le(inf, 0)
    assert tropical_le(2, 5)
    assert not tropical_le(5, 2)
    assert tropical_le(5, 5)

    arctic = SemiRing[int | float](
        null=lambda: -inf,
        choose=max,
        unit=lambda: 0,
        combine=operator.add,
    )

    arctic_le = arctic.natural_order()
    assert arctic_le(0, -inf)
    assert not arctic_le(-inf, 0)
    assert arctic_le(5, 2)
    assert not arctic_le(2, 5)
    assert arctic_le(5, 5)


def test_count() -> None:
    count: SemiRing[int] = SemiRing.count()
    check_semiring(count, (2, 3, 5), conservative=False)

    assert count.null() == 0
    assert count.choose(2, 5) == 7
    assert count.multichoose(8, 3, 1, 9) == 21
    assert count.unit() == 1
    assert count.combine(8, 3) == 24
