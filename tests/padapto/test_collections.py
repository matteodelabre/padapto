import pytest

from padapto.collections import Multiset, Record


def test_record_getattr():
    rec = Record(k1=1, k2="2", k3=(1, 2))
    assert rec.k1 == 1
    assert rec.k2 == "2"
    assert rec.k3 == (1, 2)

    with pytest.raises(AttributeError, match="record has no field 'example'"):
        rec.example  # noqa: B018


def test_record_setattr():
    rec = Record()

    with pytest.raises(NotImplementedError, match="cannot assign to field 'example'"):
        rec.example = "3"


def test_record_repr():
    assert repr(Record(k1=1, k2="2", k3=(1, 2))) in (
        "Record(k1=1, k2='2', k3=(1, 2))",
        "Record(k1=1, k3=(1, 2), k2='2')",
        "Record(k2='2', k1=1, k3=(1, 2))",
        "Record(k2='2', k3=(1, 2), k1=1)",
        "Record(k3=(1, 2), k1=1, k2='2')",
        "Record(k3=(1, 2), k2='2', k1=1)",
    )


def test_candidates_init():
    assert Multiset(()) == Multiset()
    assert Multiset(range(10)) == Multiset((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))


def test_candidates_eq_ne():
    assert Multiset((1, 2, 3, 4)) == Multiset((1, 2, 3, 4))
    assert not (Multiset((1, 2, 3, 4)) != Multiset((1, 2, 3, 4)))  # noqa: SIM202

    assert Multiset((1, 2, 3, 4)) == Multiset((4, 3, 2, 1))
    assert not (Multiset((1, 2, 3, 4)) != Multiset((4, 3, 2, 1)))  # noqa: SIM202

    assert not (Multiset((1, 2, 3, 4)) == Multiset((1, 2, 3)))  # noqa: SIM201
    assert Multiset((1, 2, 3, 4)) != Multiset((1, 2, 3))

    assert Multiset((1, 1, 2, 2)) == Multiset((1, 1, 2, 2))
    assert not (Multiset((1, 1, 2, 2)) != Multiset((1, 1, 2, 2)))  # noqa: SIM202

    assert Multiset((1, 1, 2, 2)) == Multiset((2, 2, 1, 1))
    assert not (Multiset((1, 1, 2, 2)) != Multiset((2, 2, 1, 1)))  # noqa: SIM202

    assert not (Multiset((1, 1, 2, 2)) == Multiset((1, 2, 2, 2)))  # noqa: SIM201
    assert Multiset((1, 1, 2, 2)) != Multiset((1, 2))


def test_candidates_compare():
    assert Multiset((1, 2, 3, 4)) <= Multiset((1, 2, 3, 4))
    assert Multiset((1, 2, 3, 4)) >= Multiset((1, 2, 3, 4))
    assert not (Multiset((1, 2, 3, 4)) > Multiset((1, 2, 3, 4)))
    assert not (Multiset((1, 2, 3, 4)) < Multiset((1, 2, 3, 4)))

    assert Multiset((1, 1, 2, 3)) <= Multiset((1, 1, 2, 2, 3, 4))
    assert not (Multiset((1, 1, 2, 3)) >= Multiset((1, 1, 2, 2, 3, 4)))
    assert Multiset((1, 1, 2, 3)) < Multiset((1, 1, 2, 2, 3, 4))
    assert not (Multiset((1, 1, 2, 3)) > Multiset((1, 1, 2, 2, 3, 4)))

    assert not (Multiset((1, 1, 2, 2, 3, 4)) <= Multiset((1, 1, 2, 3)))
    assert Multiset((1, 1, 2, 2, 3, 4)) >= Multiset((1, 1, 2, 3))
    assert not (Multiset((1, 1, 2, 2, 3, 4)) < Multiset((1, 1, 2, 3)))
    assert Multiset((1, 1, 2, 2, 3, 4)) > Multiset((1, 1, 2, 3))

    assert not (Multiset((1, 2, 3)) <= Multiset((2, 3, 4)))
    assert not (Multiset((1, 2, 3)) >= Multiset((2, 3, 4)))
    assert not (Multiset((1, 2, 3)) < Multiset((2, 3, 4)))
    assert not (Multiset((1, 2, 3)) > Multiset((2, 3, 4)))


def test_candidates_hash():
    assert hash(Multiset((1, 2, 3, 4))) == hash(Multiset((1, 2, 3, 4)))
    assert hash(Multiset((1, 2, 3, 4))) == hash(Multiset((1, 2, 3, 4)))
    assert hash(Multiset((1, 2, 3, 4))) == hash(Multiset((4, 3, 2, 1)))
    assert hash(Multiset((1, 1, 2, 2))) == hash(Multiset((1, 1, 2, 2)))
    assert hash(Multiset((1, 1, 2, 2))) == hash(Multiset((2, 2, 1, 1)))


def test_candidates_repr():
    assert repr(Multiset((1, 1, 2, 2))) == "Multiset((1, 1, 2, 2))"


def test_candidates_add():
    assert Multiset((1, 3, 2)) + Multiset((1, 4)) == Multiset((1, 3, 2, 1, 4))
    assert Multiset((1, 3, 2)) + Multiset((1, 4)) == Multiset((1, 4)) + Multiset(
        (1, 3, 2)
    )


def test_candidates_mul_rmul():
    assert Multiset((1, 2)) * 3 == Multiset((1, 1, 1, 2, 2, 2))
    assert 3 * Multiset((1, 2)) == Multiset((1, 1, 1, 2, 2, 2))


def test_candidates_count():
    assert Multiset((1, 2, 1, 2, 1)).count(1) == 3
    assert Multiset((1, 2, 1, 2, 1)).count(2) == 2
    assert Multiset((1, 2, 1, 2, 1)).count(3) == 0
