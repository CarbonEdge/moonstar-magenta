from moonstar_magenta.bench import rtf


def test_rtf_faster_than_realtime():
    assert rtf(10, 5) == 2.0


def test_rtf_zero_wall():
    assert rtf(10, 0) == 0.0
