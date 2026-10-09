import os

import numpy as np
import pytest

from pmve import synth


def test_note_to_midi():
    h = synth.HarmonyEngine()
    assert h.note_to_midi('A', 4) == 69
    assert h.note_to_midi('C', 4) == 60
    with pytest.raises(ValueError):
        h.note_to_midi('H', 4)


def test_7th_chord_intervals():
    h = synth.HarmonyEngine()
    chord = h.generate_7th_chord('C', 'major7', octave=4)
    assert chord['midi'] == [60, 64, 67, 71]
    assert len(chord['freqs']) == 4
    assert chord['freqs'][0] == pytest.approx(261.63, abs=0.01)
    minor = h.generate_7th_chord('A', 'minor7', octave=3)
    assert minor['midi'][1] - minor['midi'][0] == 3
    with pytest.raises(ValueError):
        h.generate_7th_chord('C', 'ninth')


def test_drop2_voicing():
    h = synth.HarmonyEngine()
    plain = h.generate_7th_chord('C', 'major7', octave=4)
    drop2 = h.generate_7th_chord('C', 'major7', octave=4, drop_2=True)
    assert drop2['midi'] == sorted(drop2['midi'])
    assert drop2['midi'] != plain['midi']


def test_synthesis_forge_shape_and_envelope():
    forge = synth.SynthesisForge(sample_rate=44100)
    wave = forge.generate(440.0, 1.0, wave_type='sine', amplitude=0.5)
    assert wave.shape == (44100,)
    assert abs(wave[0]) < 0.01  # attack starts near zero
    assert abs(wave[-1]) < 0.01  # release ends near zero
    assert np.max(np.abs(wave)) <= 0.5 + 1e-9
    with pytest.raises(ValueError):
        forge.generate(440.0, 1.0, wave_type='triangle')


def test_vocal_formant_passthrough():
    vocal = synth.VocalModel()
    sig = np.random.randn(1000)
    out = vocal.apply_vowel_formant(sig, vowel='zzz')
    assert np.array_equal(out, sig)
    colored = vocal.apply_vowel_formant(sig, vowel='ah')
    assert colored.shape == sig.shape
    assert not np.array_equal(colored, sig)


def test_mixer_buss(tmp_path):
    mixer = synth.MixerBuss()
    with pytest.raises(ValueError):
        mixer.sum_and_export([])
    a = np.ones(1000) * 0.8
    b = np.ones(500) * 0.8  # shorter stem gets padded
    out = str(tmp_path / "mix.wav")
    status = mixer.sum_and_export([a, b], out)
    assert os.path.exists(out)
    assert "mix.wav" in status
    # peak 1.6 -> normalized to 0.99
    import scipy.io.wavfile as wavfile
    _sr, data = wavfile.read(out)
    assert np.max(np.abs(data)) <= int(0.99 * 32767) + 1


def test_prosody_auditor():
    pytest.importorskip("nltk")
    try:
        auditor = synth.ProsodyAuditor()
    except Exception:
        pytest.skip("cmudict corpus unavailable")
    pattern = auditor.get_stress_pattern("hello")
    assert pattern and all(s in (0, 1) for s in pattern)
    assert auditor.get_stress_pattern("zzqzx") == [0]
    text = auditor.analyze_lyrics("hello world")
    assert "hello" in text and "world" in text
