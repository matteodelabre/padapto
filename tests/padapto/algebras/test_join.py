import operator
from collections.abc import Callable
from dataclasses import dataclass
from math import inf, prod
from typing import cast

import pytest

from padapto.algebras.join import get_subalgebras, get_subrecord, join
from padapto.algebras.signature import Signature, get_algebra_parent
from padapto.collections import Record

from .test_signature import SemiRing, check_semiring


def test_join_invalid() -> None:
    with pytest.raises(
        TypeError,
        match="join: at least one subalgebra must be provided",
    ):
        join()

    first = SemiRing[int](null=lambda: 0, choose=min, unit=lambda: 0, combine=min)
    example = Signature[int](null=lambda: 0, choose=min)

    with pytest.raises(
        TypeError,
        match=(
            "join: subalgebras for 'example' and 'first' derive "
            "from different signatures"
        ),
    ):
        join(first=first, example=example)


def test_join_plain_semirings() -> None:
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

    joined = cast(SemiRing[Record], join(left=tropical, right=integers))

    assert joined.null() == Record(left=inf, right=0)
    assert joined.unit() == Record(left=0, right=1)
    assert joined.choose(Record(left=2, right=5), Record(left=3, right=7)) == Record(
        left=min(2, 3),
        right=5 + 7,
    )
    assert joined.combine(Record(left=2, right=5), Record(left=3, right=7)) == Record(
        left=2 + 3,
        right=5 * 7,
    )

    check_semiring(
        joined,
        (Record(left=2, right=5), Record(left=3, right=7), Record(left=11, right=3)),
        conservative=False,
    )


def test_join_plain_nested():
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

    join1 = cast(SemiRing[Record], join(left=tropical, right=integers))
    join2 = cast(SemiRing[Record], join(couple=join1, single=tropical))

    assert join2.null() == Record(couple=Record(left=inf, right=0), single=inf)
    assert join2.unit() == Record(couple=Record(left=0, right=1), single=0)
    assert join2.choose(
        Record(couple=Record(left=2, right=5), single=7),
        Record(couple=Record(left=11, right=3), single=5),
    ) == Record(couple=Record(left=min(2, 11), right=3 + 5), single=min(5, 7))
    assert join2.combine(
        Record(couple=Record(left=2, right=5), single=7),
        Record(couple=Record(left=11, right=3), single=5),
    ) == Record(couple=Record(left=2 + 11, right=3 * 5), single=5 + 7)

    check_semiring(
        join2,
        (
            Record(couple=Record(left=2, right=5), single=7),
            Record(couple=Record(left=11, right=3), single=5),
            Record(couple=Record(left=0, right=7), single=3),
        ),
        conservative=False,
    )


@dataclass(frozen=True, slots=True)
class TypedRecord:
    first: int | float
    second: int


def test_join_typed_semirings() -> None:
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
        SemiRing[TypedRecord], join(TypedRecord, first=tropical, second=integers)
    )

    assert joined.null() == TypedRecord(first=inf, second=0)
    assert joined.unit() == TypedRecord(first=0, second=1)
    assert joined.choose(
        TypedRecord(first=2, second=5), TypedRecord(first=3, second=7)
    ) == TypedRecord(
        first=min(2, 3),
        second=5 + 7,
    )
    assert joined.combine(
        TypedRecord(first=2, second=5), TypedRecord(first=3, second=7)
    ) == TypedRecord(
        first=2 + 3,
        second=5 * 7,
    )

    check_semiring(
        joined,
        (
            TypedRecord(first=2, second=5),
            TypedRecord(first=3, second=7),
            TypedRecord(first=11, second=3),
        ),
        conservative=False,
    )


@dataclass(frozen=True)
class Example[T](Signature[T]):
    m1: Callable[[T, T, T], T]
    m2: Callable[[str, T, str], T]
    m3: Callable[[*tuple[str, ...]], T]
    m4: Callable[[*tuple[T, ...]], T]
    m5: Callable[[str, T, str, *tuple[T, ...]], T]


def test_join_multitypes() -> None:
    def a_m1(x: int, y: int, z: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(z, int)
        return x + y + z

    def a_m2(x: str, y: int, z: str) -> int:
        assert isinstance(x, str)
        assert isinstance(y, int)
        assert isinstance(z, str)
        return len(x) + y + len(z)

    def a_m3(*args: str) -> int:
        for arg in args:
            assert isinstance(arg, str)

        return sum(len(arg) for arg in args)

    def a_m4(*args: int) -> int:
        for arg in args:
            assert isinstance(arg, int)

        return prod(args)

    def a_m5(x: str, y: int, z: str, *args: int) -> int:
        assert isinstance(x, str)
        assert isinstance(y, int)
        assert isinstance(z, str)

        for arg in args:
            assert isinstance(arg, int)

        return len(x) + y + len(z) + sum(args)

    a = Example[int](
        null=lambda: 0,
        choose=min,
        m1=a_m1,
        m2=a_m2,
        m3=a_m3,
        m4=a_m4,
        m5=a_m5,
    )

    def b_m1(
        x: tuple[str, ...], y: tuple[str, ...], z: tuple[str, ...]
    ) -> tuple[str, ...]:
        assert isinstance(x, tuple)
        assert isinstance(y, tuple)
        assert isinstance(z, tuple)
        return x + y + z

    def b_m2(x: str, y: tuple[str, ...], z: str) -> tuple[str, ...]:
        assert isinstance(x, str)
        assert isinstance(y, tuple)
        assert isinstance(z, str)
        return (x,) + y + (z,)

    def b_m3(*args: str) -> tuple[str, ...]:
        for arg in args:
            assert isinstance(arg, str)

        return ("".join(arg for arg in args),)

    def b_m4(*args: tuple[str, ...]) -> tuple[str, ...]:
        for arg in args:
            assert isinstance(arg, tuple)

        return sum(args, start=())

    def b_m5(
        x: str, y: tuple[str, ...], z: str, *args: tuple[str, ...]
    ) -> tuple[str, ...]:
        assert isinstance(x, str)
        assert isinstance(y, tuple)
        assert isinstance(z, str)

        for arg in args:
            assert isinstance(arg, tuple)

        return (x,) + y + (z,) + sum(args, start=())

    a = Example[int](
        null=lambda: 0,
        choose=min,
        m1=a_m1,
        m2=a_m2,
        m3=a_m3,
        m4=a_m4,
        m5=a_m5,
    )
    b = Example[tuple[str, ...]](
        null=lambda: (),
        choose=operator.add,
        m1=b_m1,
        m2=b_m2,
        m3=b_m3,
        m4=b_m4,
        m5=b_m5,
    )
    joined = cast(Example[Record], join(a=a, b=b))

    assert joined.m1(
        Record(a=1, b=("a",)), Record(a=2, b=("b",)), Record(a=3, b=("c",))
    ) == Record(
        a=a.m1(1, 2, 3),
        b=b.m1(("a",), ("b",), ("c",)),
    )
    assert joined.m2("a", Record(a=1, b=("b",)), "c") == Record(
        a=a.m2("a", 1, "c"), b=b.m2("a", ("b",), "c")
    )
    assert joined.m3("a", "b", "c") == Record(
        a=a.m3("a", "b", "c"), b=b.m3("a", "b", "c")
    )
    assert joined.m4(Record(a=1, b=("a",)), Record(a=2, b=("b",))) == Record(
        a=a.m4(1, 2),
        b=b.m4(("a",), ("b",)),
    )
    assert joined.m5("a", Record(a=1, b=("a",)), "b", Record(a=2, b=("b",))) == Record(
        a=a.m5("a", 1, "b", 2),
        b=b.m5("a", ("a",), "b", ("b",)),
    )

    with pytest.raises(TypeError, match="expected 3 arguments, got 0"):
        joined.m1()  # type: ignore[call-arg,misc]

    with pytest.raises(TypeError, match="argument #1 must be of type 'Record'"):
        joined.m1(Record(a=1, b=("a",)), "a", "a")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="argument #0 must be of type 'str'"):
        joined.m2(Record(a=1, b=("a",)), "a", "a")  # type: ignore[arg-type]

    with pytest.raises(AttributeError, match="record has no field 'a'"):
        joined.m1(Record(), Record(), Record())

    with pytest.raises(TypeError, match="expected at least 3 arguments, got 2"):
        joined.m5("a", Record(a=1, b=("a",)))  # type: ignore[call-arg,misc]

    with pytest.raises(TypeError, match="argument #2 must be of type 'str'"):
        joined.m5("a", Record(a=1, b=("a",)), Record(a=1, b=("a",)))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="argument #3 must be of type 'Record'"):
        joined.m5("a", Record(a=1, b=("a",)), "a", "a")  # type: ignore[arg-type]


def test_join_metadata():
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
        SemiRing[TypedRecord], join(TypedRecord, first=tropical, second=integers)
    )
    assert get_algebra_parent(joined) == (
        "join",
        (TypedRecord,),
        {
            "first": tropical,
            "second": integers,
        },
    )


def test_get_subrecord():
    record = Record(left=Record(a=1, b=2), right="test")

    assert get_subrecord(record, "left") == record.left
    assert get_subrecord(record, "right") == record.right
    assert get_subrecord(record, "left.a") == record.left.a
    assert get_subrecord(record, "left.b") == record.left.b

    with pytest.raises(AttributeError, match="record has no field 'c'"):
        get_subrecord(record, "left.c")


def test_get_subalgebras():
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

    join0 = cast(SemiRing[Record], join(alpha=tropical, beta=integers))
    join1 = cast(SemiRing[Record], join(a=tropical, b=integers, c=join0))
    join2 = cast(SemiRing[Record], join(beta=tropical, gamma=integers))
    join3 = cast(SemiRing[Record], join(c=join2, d=integers))
    join4 = cast(SemiRing[Record], join(left=join1, right=join3))

    assert list(get_subalgebras(join4, "left")) == [("left", join1)]
    assert list(get_subalgebras(join4, "right")) == [("right", join3)]
    assert list(get_subalgebras(join4, "left", "right")) == [
        ("left", join1),
        ("right", join3),
    ]
    assert list(get_subalgebras(join4)) == []
    assert list(get_subalgebras(join4, "left.a")) == [("left.a", tropical)]
    assert list(get_subalgebras(join4, "left.b")) == [("left.b", integers)]
    assert list(get_subalgebras(join4, "left.c")) == [("left.c", join0)]
    assert list(get_subalgebras(join4, "left.c.alpha")) == [("left.c.alpha", tropical)]
    assert list(get_subalgebras(join4, "left.c.beta")) == [("left.c.beta", integers)]
    assert list(get_subalgebras(join4, "right.c")) == [("right.c", join2)]
    assert list(get_subalgebras(join4, "right.d")) == [("right.d", integers)]

    assert list(get_subalgebras(join4, "*")) == [("left", join1), ("right", join3)]
    assert list(get_subalgebras(join4, "left.*")) == [
        ("left.a", tropical),
        ("left.b", integers),
        ("left.c", join0),
    ]
    assert list(get_subalgebras(join4, "right.*")) == [
        ("right.c", join2),
        ("right.d", integers),
    ]
    assert list(get_subalgebras(join4, "left.*", "right.*")) == [
        ("left.a", tropical),
        ("left.b", integers),
        ("left.c", join0),
        ("right.c", join2),
        ("right.d", integers),
    ]
    assert list(get_subalgebras(join4, "*.c")) == [
        ("left.c", join0),
        ("right.c", join2),
    ]
    assert list(get_subalgebras(join4, "*.c.beta")) == [
        ("left.c.beta", integers),
        ("right.c.beta", tropical),
    ]
    assert list(get_subalgebras(join4, "*.*")) == [
        ("left.a", tropical),
        ("left.b", integers),
        ("left.c", join0),
        ("right.c", join2),
        ("right.d", integers),
    ]

    with pytest.raises(TypeError, match="algebra of 'left.a' is not joined"):
        list(get_subalgebras(join4, "left.a.test"))

    with pytest.raises(TypeError, match="provided algebra is not joined"):
        list(get_subalgebras(integers, "left"))

    with pytest.raises(
        AttributeError, match="'unknown' is not a field of the provided algebra"
    ):
        list(get_subalgebras(join4, "unknown"))

    with pytest.raises(AttributeError, match="'unknown' is not a field of 'left.c'"):
        list(get_subalgebras(join4, "left.c.unknown"))

    with pytest.raises(AttributeError, match="'alpha' is not a field of 'right.c'"):
        list(get_subalgebras(join4, "*.c.alpha"))
