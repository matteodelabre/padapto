import operator
from math import inf
from typing import cast

from padapto.algebras.power import limit, power
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

    sample_values: tuple[Multiset[int | float], ...] = (
        Multiset((7,)),
        Multiset((5, 7, 23)),
        Multiset((7, 13, 17)),
    )
    check_semiring(all_trop, sample_values, conservative=False)

    assert all_trop.choose(
        Multiset((2, 3, 13)),
        Multiset((3, 5, 17)),
    ) == Multiset((2, 3, 3, 5, 13, 17))

    assert all_trop.combine(
        Multiset((2, 3, 13)),
        Multiset((3, 5, 17)),
    ) == Multiset((2 + 3, 2 + 5, 2 + 17, 3 + 3, 3 + 5, 3 + 17, 13 + 3, 13 + 5, 13 + 17))


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

    sample_values: tuple[Multiset[tuple[str, ...]], ...] = (
        Multiset((("a",),)),
        Multiset((("a", "b", "c"),)),
        Multiset((("a", "b"), ("c", "d", "e"))),
    )
    check_semiring(generator, sample_values, conservative=False)

    assert generator.choose(
        Multiset((("a", "b"), ("c",))),
        Multiset((("d",), ("e", "f"))),
    ) == Multiset((("a", "b"), ("c",), ("d",), ("e", "f")))

    assert generator.combine(
        Multiset((("a", "b"), ("c",))),
        Multiset((("d",), ("e", "f"))),
    ) == Multiset((("a", "b", "d"), ("a", "b", "e", "f"), ("c", "d"), ("c", "e", "f")))


def test_power_metadata():
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    generator = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))
    assert get_algebra_metadata(generator, power) == free


def test_limit() -> None:
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    def algo(algebra: SemiRing[Multiset[tuple[str, ...]]]) -> Multiset[tuple[str, ...]]:
        return algebra.combine(
            Multiset((("a",), ("b",), ("c",))),
            algebra.choose(
                Multiset((("d",), ("e",))),
                Multiset((("f",), ("g",), ("h",))),
            ),
        )

    gen_all = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))
    all_results = algo(gen_all)

    for i in range(len(all_results)):
        gen_some = limit(gen_all, maxsize=i)
        some_results = algo(gen_some)
        assert len(some_results) == i
        assert some_results <= all_results

    assert all_results == algo(limit(gen_all, maxsize=len(all_results)))


def test_limit_metadata():
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    gen_all = cast(SemiRing[Multiset[tuple[str, ...]]], power(free))
    gen_some = limit(gen_all, maxsize=10)
    gen_some_more = limit(gen_some, maxsize=15)
    gen_some_less = limit(gen_some, maxsize=5)

    assert get_algebra_metadata(gen_some, power) == free
    assert get_algebra_metadata(gen_some_more, power) == free
    assert get_algebra_metadata(gen_some_less, power) == free
    assert get_algebra_metadata(gen_some, limit) == (gen_all, 10)
    assert get_algebra_metadata(gen_some_more, limit) == (gen_all, 10)
    assert get_algebra_metadata(gen_some_less, limit) == (gen_all, 5)
