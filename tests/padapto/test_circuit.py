from padapto.circuit import make_node, get_solution, enumerate_solutions


def test_trace_enumerate_one():
    assert (
        get_solution(
            make_node("choose")
            .add(make_node("unit1"))
            .add(make_node("unit2"))
        )
        == make_node("unit1")
    )


def test_trace_enumerate_all():
    assert list(
        enumerate_solutions(
            make_node("choose")
            .add(make_node("unit1"))
            .add(make_node("unit2"))
        )
    ) == [make_node("unit1"), make_node("unit2")]

    assert list(
        enumerate_solutions(
            make_node("combine")
            .add(
                make_node("choose")
                .add(make_node("unit1"))
                .add(make_node("unit2"))
            )
            .add(
                make_node("choose")
                .add(make_node("unit2"))
                .add(make_node("unit3"))
            )
        )
    ) == [
        make_node("combine").add(make_node("unit1")).add(make_node("unit2")),
        make_node("combine").add(make_node("unit1")).add(make_node("unit3")),
        make_node("combine").add(make_node("unit2")).add(make_node("unit2")),
        make_node("combine").add(make_node("unit2")).add(make_node("unit3")),
    ]
