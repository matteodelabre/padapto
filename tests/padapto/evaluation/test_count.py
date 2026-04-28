from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from padapto.evaluation.count import count
from padapto.signature import Signature


@dataclass(frozen=True)
class OutSemiRing[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[str, T, str, T], T]


def test_count() -> None:
    counter = cast(OutSemiRing[int], count(OutSemiRing))

    assert counter.null() == 0
    assert counter.choose(3, 7) == 10
    assert counter.multichoose(3, 7, 1, 4) == 15
    assert counter.unit() == 1
    assert counter.combine("a", 3, "b", 7) == 21
