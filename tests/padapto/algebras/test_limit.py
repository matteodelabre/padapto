from typing import cast

from padapto.algebras.limit import limit
from padapto.algebras.power import power
from padapto.algebras.signature import get_algebra_metadata
from padapto.collections import Multiset

from .test_signature import SemiRing


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

    gen_all = cast(SemiRing[Multiset[tuple[str, ...]]], free | power())
    all_results = algo(gen_all)

    for i in range(len(all_results)):
        gen_some = gen_all | limit(maxsize=i)
        some_results = algo(gen_some)
        assert len(some_results) == i
        assert some_results <= all_results

    assert all_results == algo(gen_all | limit(maxsize=len(all_results)))


def test_limit_metadata():
    free = SemiRing[tuple[str, ...] | None](
        null=lambda: None,
        choose=lambda x, y: x if x is not None else y,
        unit=lambda: (),
        combine=lambda x, y: x + y if x is not None and y is not None else None,
    )

    gen_all = cast(SemiRing[Multiset[tuple[str, ...]]], free | power())
    gen_some = gen_all | limit(maxsize=10)
    gen_some_more = gen_all | limit(maxsize=10) | limit(maxsize=15)
    gen_some_less = gen_all | limit(maxsize=10) | limit(maxsize=5)

    assert get_algebra_metadata(gen_some, power) == free
    assert get_algebra_metadata(gen_some_more, power) == free
    assert get_algebra_metadata(gen_some_less, power) == free
    assert get_algebra_metadata(gen_some, limit) == (gen_all, 10)
    assert get_algebra_metadata(gen_some_more, limit) == (gen_all, 10)
    assert get_algebra_metadata(gen_some_less, limit) == (gen_all, 5)
