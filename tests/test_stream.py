from moonstar_magenta.stream import parse_schedule, prompt_at, underruns


def test_schedule_switches():
    s = parse_schedule(["8:techno", "0:piano"])
    assert prompt_at(s, 0.0) == "piano"
    assert prompt_at(s, 7.9) == "piano"
    assert prompt_at(s, 8.0) == "techno"


def test_underruns():
    assert underruns([0.5, 1.2, 0.9, 1.0], 1.0) == 1


def test_parse_spec_weights_and_audio():
    from moonstar_magenta.blend import parse_spec
    assert parse_spec("funk") == [(1.0, "funk")]
    assert parse_spec("0.7*file:a.wav | 0.3*funky bass") == [(0.7, "file:a.wav"), (0.3, "funky bass")]


def test_blend_normalises_weights():
    import numpy as np
    from moonstar_magenta.blend import blend
    out = blend([(3.0, np.array([1.0, 0.0])), (1.0, np.array([0.0, 1.0]))])
    assert np.allclose(out, [0.75, 0.25])


def test_live_helpers():
    import struct

    import numpy as np

    from moonstar_magenta.live import sleep_needed, to_pcm16, wav_header
    assert len(wav_header()) == 44 and wav_header()[:4] == b"RIFF"
    assert struct.unpack("<I", wav_header()[24:28])[0] == 48000
    assert sleep_needed(5.0, 1.0, 2.0) == 2.0 and sleep_needed(1.0, 1.0, 2.0) == 0.0
    assert len(to_pcm16(np.zeros((25, 2), dtype=np.float32))) == 100
