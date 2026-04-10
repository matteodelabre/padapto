from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from random import Random

from padapto.algebras.cost import add_optimizer, boltzmann
from padapto.algebras.counter import counter
from padapto.algebras.signature import Signature
from padapto.circuit import (
    enumerate_solutions,
    eval,
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


def _weighted_sample(circuit, alg, repeats):
    weights = eval_inside(circuit, alg)
    gen = Random(42)
    outcomes = Counter()

    for _ in range(repeats):
        outcomes[sample(circuit, gen, weights)] += 1

    return outcomes


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
    sols = set(enumerate_solutions(grid))
    outcomes = _weighted_sample(grid, weighting, repeats=len(sols) * 1000)
    _assert_distrib_uniform(outcomes, sols, tol=0.1)


def test_circuit_sample_paren_uniform():
    paren = _make_paren(4)
    weighting = counter(ParenSignature)
    sols = set(enumerate_solutions(paren))
    outcomes = _weighted_sample(paren, weighting, repeats=len(sols) * 1000)
    _assert_distrib_uniform(outcomes, sols, tol=0.1)


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
    opt_outcomes = _weighted_sample(grid, opt_weighting, repeats=len(opt_sols) * 1000)
    _assert_distrib_uniform(opt_outcomes, opt_sols, tol=0.1)
    _assert_distrib_none_out(opt_outcomes, opt_sols, bound=10)

    # # High temperature means uniform sampling among all solutions
    all_sols = set(enumerate_solutions(grid))
    all_weighting = boltzmann(GridSignature, temperature=100, **operators)
    all_outcomes = _weighted_sample(grid, all_weighting, repeats=len(all_sols) * 1000)
    _assert_distrib_uniform(all_outcomes, all_sols, tol=0.1)
    _assert_distrib_none_out(all_outcomes, all_sols, bound=0)

    # Intermediate temperature means less costly solutions are sampled more often
    med_weighting = boltzmann(GridSignature, temperature=1, **operators)
    med_outcomes = _weighted_sample(grid, med_weighting, repeats=len(all_sols) * 1000)
    med_most_common = [eval(sol, cost_eval) for sol, _ in med_outcomes.most_common()]
    assert all_sols == set(med_outcomes.keys())
    assert med_most_common == sorted([eval(sol, cost_eval) for sol in all_sols])


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
