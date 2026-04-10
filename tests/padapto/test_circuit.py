from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from random import Random

from padapto.algebras.signature import Signature
from padapto.algebras.counter import counter
from padapto.circuit import (
    enumerate_solutions,
    eval_inside,
    get_solution,
    make_node,
    render,
    sample,
)


@dataclass(frozen=True)
class GridSignature[T](Signature[T]):
    unit: Callable[[], T]
    left: Callable[[int, T, int], T]
    up: Callable[[int, T, int], T]
    diag: Callable[[int, T, int], T]


def _make_grid(size):
    grid = [[None] * size for _ in range(size)]

    for i, j in product(range(size), range(size)):
        node = make_node("unit") if i == 0 and j == 0 else make_node("choose")

        if i >= 1:
            node = node.add(make_node("left", (i, j)).add(grid[i - 1][j]))

        if j >= 1:
            node = node.add(make_node("up", (i, j)).add(grid[i][j - 1]))

        if i >= 1 and j >= 1:
            node = node.add(make_node("diag", (i, j)).add(grid[i - 1][j - 1]))

        grid[i][j] = node

    return grid[size - 1][size - 1]


@dataclass(frozen=True)
class ParenSignature[T](Signature[T]):
    unit: Callable[[], T]
    combine: Callable[[T, T, int, int, int], T]


def _make_paren(size):
    cells = [[None] * (size + 1) for _ in range(size)]

    for s in range(1, size + 1):
        for i in range(size - s + 1):
            if s == 1:
                cells[i][s] = make_node("unit")
            else:
                cells[i][s] = make_node("choose")

                for k in range(1, s):
                    cells[i][s] = cells[i][s].add(
                        make_node("combine", (i, i + k, i + s))
                        .add(cells[i][k])
                        .add(cells[i + k][s - k])
                    )

                if len(cells[i][s].edges) == 1:
                    cells[i][s] = cells[i][s].edges[0].node

    return cells[0][size]


def test_circuit_get_solution():
    assert get_solution(
        make_node("choose").add(make_node("unit1")).add(make_node("unit2"))
    ) == make_node("unit1")


def test_circuit_get_solution_grid():
    # Even with a large circuit, getting a single solution should take reasonable time.
    # Here, the grid circuit of size 30 contains 1 682 471 873 186 160 624 243 possible
    # paths (29th term of OEIS A001850), yet `get_solution` should still complete in a
    # fraction of a second
    size = 20
    grid = _make_grid(size)
    res = make_node("unit")

    for i in range(1, size):
        res = make_node("up", (0, i)).add(res)

    for i in range(1, size):
        res = make_node("left", (i, size - 1)).add(res)

    sol = get_solution(grid)
    assert res == sol


def test_circuit_get_solution_paren():
    # Same here but with a circuit containing combination nodes. The number of solutions
    # in this case is 1 002 242 216 651 368 (29th Catalan number)
    size = 30
    paren = _make_paren(size)
    res = make_node("unit")

    for i in range(size - 1, 0, -1):
        res = make_node("combine", (i - 1, i, size)).add(make_node("unit")).add(res)

    sol = get_solution(paren)
    assert res == sol


def test_circuit_enumerate_all():
    assert list(
        enumerate_solutions(
            make_node("choose").add(make_node("unit1")).add(make_node("unit2"))
        )
    ) == [make_node("unit1"), make_node("unit2")]

    assert list(
        enumerate_solutions(
            make_node("combine")
            .add(make_node("choose").add(make_node("unit1")).add(make_node("unit2")))
            .add(make_node("choose").add(make_node("unit2")).add(make_node("unit3")))
        )
    ) == [
        make_node("combine").add(make_node("unit1")).add(make_node("unit2")),
        make_node("combine").add(make_node("unit1")).add(make_node("unit3")),
        make_node("combine").add(make_node("unit2")).add(make_node("unit2")),
        make_node("combine").add(make_node("unit2")).add(make_node("unit3")),
    ]

    assert list(enumerate_solutions(_make_grid(2))) == [
        make_node("left", (1, 1)).add(make_node("up", (0, 1)).add(make_node("unit"))),
        make_node("up", (1, 1)).add(make_node("left", (1, 0)).add(make_node("unit"))),
        make_node("diag", (1, 1)).add(make_node("unit")),
    ]

    assert list(enumerate_solutions(_make_paren(3))) == [
        make_node("combine", (0, 1, 3))
        .add(make_node("unit"))
        .add(
            make_node("combine", (1, 2, 3))
            .add(make_node("unit"))
            .add(make_node("unit"))
        ),
        make_node("combine", (0, 2, 3))
        .add(
            make_node("combine", (0, 1, 2))
            .add(make_node("unit"))
            .add(make_node("unit"))
        )
        .add(make_node("unit")),
    ]


def _assert_sample_circuit_distrib(circuit, alg, ampl, tol):
    weights = eval_inside(circuit, alg)
    total = weights[id(circuit)]

    gen = Random(42)
    outcomes = Counter()

    for _ in range(total * ampl):
        outcomes[sample(circuit, gen, weights)] += 1

    assert len(outcomes) == total
    assert all(ampl * (1 - tol) <= occ <= ampl * (1 + tol) for occ in outcomes.values())


def test_circuit_sample_uniform_grid():
    grid = _make_grid(4)
    _assert_sample_circuit_distrib(grid, counter(GridSignature), ampl=1000, tol=0.1)


def test_circuit_sample_uniform_paren():
    paren = _make_paren(4)
    _assert_sample_circuit_distrib(paren, counter(ParenSignature), ampl=1000, tol=0.1)


def test_circuit_render():
    assert (
        render(
            make_node("combine")
            .add(make_node("choose").add(make_node("unit1")).add(make_node("unit2")))
            .add(make_node("choose").add(make_node("unit2")).add(make_node("unit3")))
        )
        == """\
digraph {
0 [shape="box", style="rounded", ordering="out", label="combine"]
0 -> 1
0 -> 2
1 [label="⊕", shape="none", width="0", height="0"]
1 -> 3
1 -> 4
3 [shape="box", style="rounded", ordering="out", label="unit1"]
4 [shape="box", style="rounded", ordering="out", label="unit2"]
2 [label="⊕", shape="none", width="0", height="0"]
2 -> 5
2 -> 6
5 [shape="box", style="rounded", ordering="out", label="unit2"]
6 [shape="box", style="rounded", ordering="out", label="unit3"]
}\
"""
    )
