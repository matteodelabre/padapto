from collections.abc import Callable
from dataclasses import dataclass

from padapto.algebras.counter import counter
from padapto.algebras.signature import Signature


@dataclass(frozen=True)
class OutSemiRing[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[str, T, str, T], T]


def test_counter() -> None:
    out_counter: OutSemiRing[int] = counter(OutSemiRing)

    assert out_counter.null() == 0
    assert out_counter.choose(3, 7) == 10
    assert out_counter.multichoose(3, 7, 1, 4) == 15
    assert out_counter.unit() == 1
    assert out_counter.combine("a", 3, "b", 7) == 21
