from moonstar_magenta.stream import parse_schedule, prompt_at, underruns


def test_schedule_switches():
    s = parse_schedule(["8:techno", "0:piano"])
    assert prompt_at(s, 0.0) == "piano"
    assert prompt_at(s, 7.9) == "piano"
    assert prompt_at(s, 8.0) == "techno"


def test_underruns():
    assert underruns([0.5, 1.2, 0.9, 1.0], 1.0) == 1
