from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from random import Random

from sowing import traversal

from padapto.algebras.cost import add_optimizer, boltzmann
from padapto.algebras.counter import counter
from padapto.algebras.signature import Signature
from padapto.circuit import (
    enumerate_solutions,
    eval,
    eval_inside,
    eval_outside,
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
    unit: Callable[[int], T]
    combine: Callable[[T, T, int, int, int], T]


def _make_paren(size):
    cells = [[None] * (size + 1) for _ in range(size)]

    for s in range(1, size + 1):
        for i in range(size - s + 1):
            if s == 1:
                cells[i][s] = make_node("unit", (i,))
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


def test_paren_non_redundant():
    paren = _make_paren(9)
    keys = list(
        cursor.node.data.args
        for cursor in traversal.depth(paren, unique="id")
        if not cursor.node.data.is_choose()
    )
    assert len(keys) == len(set(keys))


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
    res = make_node("unit", (size - 1,))

    for i in range(size - 1, 0, -1):
        res = (
            make_node("combine", (i - 1, i, size))
            .add(make_node("unit", (i - 1,)))
            .add(res)
        )

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
        .add(make_node("unit", (0,)))
        .add(
            make_node("combine", (1, 2, 3))
            .add(make_node("unit", (1,)))
            .add(make_node("unit", (2,)))
        ),
        make_node("combine", (0, 2, 3))
        .add(
            make_node("combine", (0, 1, 2))
            .add(make_node("unit", (0,)))
            .add(make_node("unit", (1,)))
        )
        .add(make_node("unit", (2,))),
    ]


def _weighted_sample(circuit, alg, repeats):
    weights = eval_inside(circuit, alg)
    gen = Random(42)

    for _ in range(repeats):
        yield sample(circuit, gen, weights)


def _assert_distrib_uniform(outcomes, sols, tol):
    mean = outcomes.total() / len(sols)
    assert all(
        mean * (1 - tol) <= occ <= mean * (1 + tol)
        for sol, occ in outcomes.items()
        if sol in sols
    )


def _assert_distrib_none_out(outcomes, sols, bound):
    assert all(occ <= bound for sol, occ in outcomes.items() if sol not in sols)


def test_circuit_sample_grid_uniform():
    grid = _make_grid(4)
    weighting = counter(GridSignature)
    all_sols = set(enumerate_solutions(grid))
    sample_sols = _weighted_sample(grid, weighting, repeats=len(all_sols) * 1000)
    _assert_distrib_uniform(Counter(sample_sols), all_sols, tol=0.1)


def test_circuit_sample_paren_uniform():
    paren = _make_paren(4)
    weighting = counter(ParenSignature)
    all_sols = set(enumerate_solutions(paren))
    sample_sols = _weighted_sample(paren, weighting, repeats=len(all_sols) * 1000)
    _assert_distrib_uniform(Counter(sample_sols), all_sols, tol=0.1)


def test_circuit_sample_grid_boltzmann():
    grid = _make_grid(4)
    operators = {
        "left": lambda i, j: abs(i - j),
        "up": lambda i, j: abs(i - j),
        "diag": lambda i, j: abs(i - j) + 1,
    }
    cost_eval = add_optimizer(GridSignature, **operators)

    # Low temperature means uniform sampling among optimal solutions only
    opt_sols = {sol for sol in enumerate_solutions(grid) if eval(sol, cost_eval) == 3}
    opt_weighting = boltzmann(GridSignature, temperature=0.1, **operators)
    opt_sample = _weighted_sample(grid, opt_weighting, repeats=len(opt_sols) * 1000)
    opt_outcomes = Counter(opt_sample)
    _assert_distrib_uniform(opt_outcomes, opt_sols, tol=0.1)
    _assert_distrib_none_out(opt_outcomes, opt_sols, bound=10)

    # High temperature means uniform sampling among all solutions
    all_sols = set(enumerate_solutions(grid))
    all_weighting = boltzmann(GridSignature, temperature=100, **operators)
    all_sample = _weighted_sample(grid, all_weighting, repeats=len(all_sols) * 1000)
    all_outcomes = Counter(all_sample)
    _assert_distrib_uniform(all_outcomes, all_sols, tol=0.1)
    _assert_distrib_none_out(all_outcomes, all_sols, bound=0)

    # Intermediate temperature means less costly solutions are sampled more often
    med_weighting = boltzmann(GridSignature, temperature=1, **operators)
    med_sample = _weighted_sample(grid, med_weighting, repeats=len(all_sols) * 1000)
    med_outcomes = Counter(med_sample)
    med_most_common = [eval(sol, cost_eval) for sol, _ in med_outcomes.most_common()]
    assert all_sols == set(med_outcomes.keys())
    assert med_most_common == sorted([eval(sol, cost_eval) for sol in all_sols])


def test_circuit_outside_count_paren():
    paren = _make_paren(9)
    weighting = counter(ParenSignature)
    inside = eval_inside(paren, weighting)
    outside = eval_outside(paren, weighting, inside)

    # `inside[node] * outside[node]` should equal the number of solutions
    # containing a given node
    expected = {
        cursor.node.data.args: inside[id(cursor.node)] * outside[id(cursor.node)]
        for cursor in traversal.depth(paren, unique="id")
        if not cursor.node.data.is_choose()
    }

    actual = Counter(
        cursor.node.data.args
        for sol in enumerate_solutions(paren)
        for cursor in traversal.depth(sol)
        if not cursor.node.data.is_choose()
    )

    assert expected == actual


def _assert_sample_expected_subcounts(circuit, alg, repeats, tol):
    inside = eval_inside(circuit, alg)
    outside = eval_outside(circuit, alg, inside)
    root_val = inside[id(circuit)]

    expected = {
        cursor.node.data.args: inside[id(cursor.node)]
        * outside[id(cursor.node)]
        * repeats
        / root_val
        for cursor in traversal.depth(circuit, unique="id")
        if not cursor.node.data.is_choose()
    }

    actual = Counter(
        cursor.node.data.args
        for sol in _weighted_sample(circuit, alg, repeats=repeats)
        for cursor in traversal.depth(sol)
        if not cursor.node.data.is_choose()
    )

    margin = repeats * tol

    for key in set(expected) | set(actual):
        assert expected[key] - margin <= actual[key] <= expected[key] + margin


def test_circuit_outside_sample_paren():
    paren = _make_paren(7)
    operators = {
        "unit": lambda _: 0,
        "combine": lambda i, j, k: abs(i - j) + abs(j - k),
    }

    opt_weighting = boltzmann(ParenSignature, temperature=0.1, **operators)
    _assert_sample_expected_subcounts(paren, opt_weighting, repeats=10_000, tol=0.01)

    all_weighting = boltzmann(ParenSignature, temperature=100, **operators)
    _assert_sample_expected_subcounts(paren, all_weighting, repeats=10_000, tol=0.01)

    med_weighting = boltzmann(ParenSignature, temperature=1, **operators)
    _assert_sample_expected_subcounts(paren, med_weighting, repeats=10_000, tol=0.01)


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
