from math import comb
from string import ascii_lowercase

from sowing import Hedge, Node

from padapto.structure.pattern import (
    Empty,
    Item,
    Range,
    Subseq,
    Subset,
    Term,
    Tree,
    Var,
    chain,
)


def _assert_eq_iterables(it1, it2):
    for el1, el2 in zip(it1, it2, strict=True):
        assert el1 == el2


def test_pattern_var():
    _assert_eq_iterables(Var().match(42), ({},))
    _assert_eq_iterables(Var(value=12).match(42), ())
    _assert_eq_iterables(Var("X").match(42), ({"X": 42},))
    _assert_eq_iterables(Var("X", 42).match(42), ({"X": 42},))


def test_pattern_empty_item():
    _assert_eq_iterables(Empty().match(()), ({},))
    _assert_eq_iterables(Empty().match("abc"), ())

    _assert_eq_iterables(Empty().match([]), ({},))
    _assert_eq_iterables(Empty().match(list(range(3))), ())

    _assert_eq_iterables(Empty().match(Hedge()), ({},))
    _assert_eq_iterables(Empty().match(Hedge.of(Node("a"))), ())


def test_pattern_item():
    _assert_eq_iterables(Item(Var("X")).match(()), ())
    _assert_eq_iterables(Item(Var("X")).match("abc"), ())
    _assert_eq_iterables(Item(Var("X")).match("a"), ({"X": "a"},))
    _assert_eq_iterables(Item(Var("X"), Var()).match("abc"), ({"X": "a"},))

    _assert_eq_iterables(Item(Var("X")).match([]), ())
    _assert_eq_iterables(Item(Var("X")).match(list(range(3))), ())
    _assert_eq_iterables(Item(Var("X")).match([0]), ({"X": 0},))
    _assert_eq_iterables(Item(Var("X"), Var()).match(list(range(3))), ({"X": 0},))

    _assert_eq_iterables(Item(Var("X")).match(Hedge()), ())
    _assert_eq_iterables(Item(Var("X")).match(Hedge.of(Node("a"), Node("b"))), ())
    _assert_eq_iterables(
        Item(Var("X")).match(Hedge.of(Node("a"))), ({"X": Node("a").unzip()},)
    )
    _assert_eq_iterables(
        Item(Var("X"), Var()).match(Hedge.of(Node("a"), Node("b"))),
        ({"X": Node().add(Node("a")).add(Node("b")).unzip().down(0)},),
    )


def test_pattern_chain():
    assert chain(
        Subset(Var("X")),
        Item(Var("Y")),
        Subseq(Var("Z")),
    ) == Subset(Var("X"), rest=Item(Var("Y"), rest=Subseq(Var("Z"))))


def test_pattern_subseq():
    _assert_eq_iterables(Subseq(Var("X")).match("abc"), ({"X": "abc"},))

    _assert_eq_iterables(
        Subseq(Var("X"), rest=Subseq(Var("Y"))).match("abc"),
        (
            {"X": "", "Y": "abc"},
            {"X": "a", "Y": "bc"},
            {"X": "ab", "Y": "c"},
            {"X": "abc", "Y": ""},
        ),
    )

    _assert_eq_iterables(
        Subseq(Var("X"), rest=Subseq(Var("Y"))).match(list(range(3))),
        (
            {"X": [], "Y": [0, 1, 2]},
            {"X": [0], "Y": [1, 2]},
            {"X": [0, 1], "Y": [2]},
            {"X": [0, 1, 2], "Y": []},
        ),
    )

    hedge = Hedge.of(*(Node(n) for n in range(3)))
    _assert_eq_iterables(
        Subseq(Var("X"), rest=Subseq(Var("Y"))).match(hedge),
        (
            {"X": Hedge(), "Y": hedge},
            {"X": hedge[:1], "Y": hedge[1:]},
            {"X": hedge[:2], "Y": hedge[2:]},
            {"X": hedge, "Y": Hedge()},
        ),
    )


def test_pattern_subseq_item():
    _assert_eq_iterables(
        Subseq(Var("X"), rest=Item(Var(value="b"), rest=Subseq(Var("Y")))).match(
            "aabbcc"
        ),
        (
            {"X": "aa", "Y": "bcc"},
            {"X": "aab", "Y": "cc"},
        ),
    )


def test_pattern_subseq_size():
    _assert_eq_iterables(Subseq(Var("X"), size=2).match("abc"), ())
    _assert_eq_iterables(Subseq(Var("X"), size=Range(0, 2)).match("abc"), ())
    _assert_eq_iterables(Subseq(Var("X"), size=3).match("abc"), ({"X": "abc"},))
    _assert_eq_iterables(Subseq(Var("X"), size=Range(2)).match("abc"), ({"X": "abc"},))
    _assert_eq_iterables(Subseq(Var("X"), size=Range(3)).match("abc"), ({"X": "abc"},))
    _assert_eq_iterables(Subseq(Var("X"), size=4).match("abc"), ())
    _assert_eq_iterables(Subseq(Var("X"), size=Range(4)).match("abc"), ())

    _assert_eq_iterables(
        Subseq(Var("X"), size=2, rest=Subseq(Var("Y"))).match("abc"),
        ({"X": "ab", "Y": "c"},),
    )
    _assert_eq_iterables(
        Subseq(Var("X"), size=4, rest=Subseq(Var("Y"))).match("abc"),
        (),
    )

    _assert_eq_iterables(
        Subseq(Var("X"), size=Range(1), rest=Var("Y")).match("abc"),
        (
            {"X": "a", "Y": "bc"},
            {"X": "ab", "Y": "c"},
            {"X": "abc", "Y": ""},
        ),
    )
    _assert_eq_iterables(
        Subseq(Var("X"), size=Range(0, 2), rest=Var("Y")).match("abc"),
        (
            {"X": "", "Y": "abc"},
            {"X": "a", "Y": "bc"},
        ),
    )
    _assert_eq_iterables(
        Subseq(Var("X"), size=Range(0, 20), rest=Var("Y")).match("abc"),
        (
            {"X": "", "Y": "abc"},
            {"X": "a", "Y": "bc"},
            {"X": "ab", "Y": "c"},
            {"X": "abc", "Y": ""},
        ),
    )
    _assert_eq_iterables(
        Subseq(Var("X"), size=Range(step=2), rest=Var("Y")).match("abc"),
        (
            {"X": "", "Y": "abc"},
            {"X": "ab", "Y": "c"},
        ),
    )


def test_pattern_subseq_count():
    for k in range(1, 5):
        for n in range(15):
            pat = chain(*(Subseq(Var(str(i))) for i in range(k)))
            it = pat.match(ascii_lowercase[:n])
            assigns = set(tuple(assign.values()) for assign in it)
            assert len(assigns) == comb(n + k - 1, k - 1)


def test_pattern_subset():
    _assert_eq_iterables(Subset(Var("X")).match("abc"), ({"X": "abc"},))

    _assert_eq_iterables(
        Subset(Var("X"), rest=Subset(Var("Y"))).match("abc"),
        (
            {"X": (), "Y": ("a", "b", "c")},
            {"X": ("a",), "Y": ("b", "c")},
            {"X": ("b",), "Y": ("a", "c")},
            {"X": ("c",), "Y": ("a", "b")},
            {"X": ("a", "b"), "Y": ("c",)},
            {"X": ("a", "c"), "Y": ("b",)},
            {"X": ("b", "c"), "Y": ("a",)},
            {"X": ("a", "b", "c"), "Y": ()},
        ),
    )

    _assert_eq_iterables(
        Subset(Var("X"), rest=Subset(Var("Y"))).match(list(range(3))),
        (
            {"X": (), "Y": (0, 1, 2)},
            {"X": (0,), "Y": (1, 2)},
            {"X": (1,), "Y": (0, 2)},
            {"X": (2,), "Y": (0, 1)},
            {"X": (0, 1), "Y": (2,)},
            {"X": (0, 2), "Y": (1,)},
            {"X": (1, 2), "Y": (0,)},
            {"X": (0, 1, 2), "Y": ()},
        ),
    )

    hedge = Hedge.of(*(Node(n) for n in range(3)))
    _assert_eq_iterables(
        Subset(Var("X"), rest=Subset(Var("Y"))).match(hedge),
        (
            {"X": (), "Y": (hedge[0], hedge[1], hedge[2])},
            {"X": (hedge[0],), "Y": (hedge[1], hedge[2])},
            {"X": (hedge[1],), "Y": (hedge[0], hedge[2])},
            {"X": (hedge[2],), "Y": (hedge[0], hedge[1])},
            {"X": (hedge[0], hedge[1]), "Y": (hedge[2],)},
            {"X": (hedge[0], hedge[2]), "Y": (hedge[1],)},
            {"X": (hedge[1], hedge[2]), "Y": (hedge[0],)},
            {"X": (hedge[0], hedge[1], hedge[2]), "Y": ()},
        ),
    )


def test_pattern_subset_count():
    for k in range(1, 4):
        for n in range(10):
            pat = chain(*(Subset(Var(str(i))) for i in range(k)))
            it = pat.match(ascii_lowercase[:n])
            assigns = set(tuple(assign.values()) for assign in it)
            assert len(assigns) == k**n


def test_pattern_subset_subseq_item():
    _assert_eq_iterables(
        Subset(Item(Var("X")), rest=Var()).match("abc"),
        (
            {"X": "a"},
            {"X": "b"},
            {"X": "c"},
        ),
    )
    _assert_eq_iterables(
        Subset(Item(Var("X")), rest=Var()).match(list(range(3))),
        (
            {"X": 0},
            {"X": 1},
            {"X": 2},
        ),
    )
    _assert_eq_iterables(
        Subset(Subseq(Var("X"), rest=Var("Y")), rest=Var("Z")).match(list(range(3))),
        (
            {"X": (), "Y": (), "Z": (0, 1, 2)},
            {"X": (), "Y": (0,), "Z": (1, 2)},
            {"X": (0,), "Y": (), "Z": (1, 2)},
            {"X": (), "Y": (1,), "Z": (0, 2)},
            {"X": (1,), "Y": (), "Z": (0, 2)},
            {"X": (), "Y": (2,), "Z": (0, 1)},
            {"X": (2,), "Y": (), "Z": (0, 1)},
            {"X": (), "Y": (0, 1), "Z": (2,)},
            {"X": (0,), "Y": (1,), "Z": (2,)},
            {"X": (0, 1), "Y": (), "Z": (2,)},
            {"X": (), "Y": (0, 2), "Z": (1,)},
            {"X": (0,), "Y": (2,), "Z": (1,)},
            {"X": (0, 2), "Y": (), "Z": (1,)},
            {"X": (), "Y": (1, 2), "Z": (0,)},
            {"X": (1,), "Y": (2,), "Z": (0,)},
            {"X": (1, 2), "Y": (), "Z": (0,)},
            {"X": (), "Y": (0, 1, 2), "Z": ()},
            {"X": (0,), "Y": (1, 2), "Z": ()},
            {"X": (0, 1), "Y": (2,), "Z": ()},
            {"X": (0, 1, 2), "Y": (), "Z": ()},
        ),
    )


def test_pattern_term():
    _assert_eq_iterables(Term(Var("X")).match(42), ({"X": 42},))
    _assert_eq_iterables(
        Term(Var("X"), rest=Var()).match(6),
        (
            {"X": 0},
            {"X": 1},
            {"X": 2},
            {"X": 3},
            {"X": 4},
            {"X": 5},
            {"X": 6},
        ),
    )


def test_pattern_term_span():
    _assert_eq_iterables(Term(Var("X"), Range(2, 8)).match(42), ())
    _assert_eq_iterables(Term(Var("X"), Range(30, 50)).match(42), ({"X": 42},))
    _assert_eq_iterables(Term(Var("X"), Range(70, 73)).match(42), ())
    _assert_eq_iterables(
        Term(Var("X"), Range(10, 21, 3), rest=Var()).match(42),
        (
            {"X": 10},
            {"X": 13},
            {"X": 16},
            {"X": 19},
        ),
    )


def test_pattern_term_count():
    for k in range(1, 5):
        for n in range(15):
            pat = chain(*(Term(Var(str(i))) for i in range(k)))
            it = pat.match(n)
            assigns = set(tuple(assign.values()) for assign in it)
            assert len(assigns) == comb(n + k - 1, k - 1)


def test_pattern_tree():
    tree = (
        Node("a")
        .add(Node("l"))
        .add(Node("b").add(Node("e")).add(Node("f")), data="ab")
        .add(Node("c"), data="ac")
        .add(Node("d"))
    )
    cursor = tree.unzip()

    _assert_eq_iterables(
        Tree(parent=Var("X")).match(cursor),
        (),
    )
    _assert_eq_iterables(
        Tree(parent=Var("X")).match(cursor.down(1).down(1)),
        ({"X": cursor.down(1)},),
    )
    _assert_eq_iterables(Tree().match(cursor), ({},))
    _assert_eq_iterables(Tree().match(cursor.down(1).down(1)), ({},))

    _assert_eq_iterables(
        Tree(children=Var("X")).match(cursor.down(1)),
        ({"X": Hedge(cursor.down(1).down(), breadth=2)},),
    )
    _assert_eq_iterables(
        Tree(children=Var("X")).match(cursor.down(2)), ({"X": Hedge()},)
    )

    _assert_eq_iterables(
        Tree(siblings=Var("X")).match(cursor.down(0)),
        ({"X": (cursor.down(1), cursor.down(2), cursor.down(3))},),
    )
    _assert_eq_iterables(
        Tree(siblings=Var("X")).match(cursor.down(1)),
        ({"X": (cursor.down(0), cursor.down(2), cursor.down(3))},),
    )
    _assert_eq_iterables(
        Tree(siblings=Var("X")).match(cursor.down(2)),
        ({"X": (cursor.down(0), cursor.down(1), cursor.down(3))},),
    )
    _assert_eq_iterables(
        Tree(siblings=Var("X")).match(cursor.down(3)),
        ({"X": (cursor.down(0), cursor.down(1), cursor.down(2))},),
    )

    _assert_eq_iterables(
        Tree(
            node=Var("node"),
            edge=Var("edge"),
            parent=Var("parent"),
            children=Var("children"),
            siblings=Var("siblings"),
        ).match(cursor.down(1)),
        (
            {
                "node": "b",
                "edge": "ab",
                "parent": cursor,
                "children": Hedge(cursor.down(1).down(0), breadth=2),
                "siblings": (cursor.down(0), cursor.down(2), cursor.down(3)),
            },
        ),
    )
