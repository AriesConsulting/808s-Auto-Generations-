"""Music synthesis toolkit: harmony, waveform generation, vocal formants, mixing.

Extracted from the Sovereign Production Engine notebook. Pure DSP on numpy;
only ProsodyAuditor needs the optional nltk cmudict corpus.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal


class HarmonyEngine:
    """Chord construction: note names to MIDI numbers to frequencies."""

    def __init__(self) -> None:
        self.notes = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                      'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    def note_to_midi(self, note_name: str, octave: int) -> int:
        if note_name not in self.notes:
            raise ValueError(f"Invalid note: {note_name}")
        return self.notes.index(note_name) + (octave + 1) * 12

    def generate_7th_chord(self, root_note: str, chord_type: str = 'major7',
                           octave: int = 4, drop_2: bool = False) -> Dict:
        intervals = {
            'major7': [0, 4, 7, 11],
            'minor7': [0, 3, 7, 10],
            'dominant7': [0, 4, 7, 10],
            'm7b5': [0, 3, 6, 10],
        }
        if chord_type not in intervals:
            raise ValueError(f"Unknown chord_type: {chord_type}")

        root_midi = self.note_to_midi(root_note, octave)
        midi_notes = [root_midi + i for i in intervals[chord_type]]

        if drop_2 and len(midi_notes) >= 4:
            sorted_notes = sorted(midi_notes)
            sorted_notes[-2] -= 12
            midi_notes = sorted(sorted_notes)

        freqs = [440 * 2 ** ((m - 69) / 12) for m in midi_notes]
        return {'midi': midi_notes, 'freqs': freqs}


class SynthesisForge:
    """Oscillator with an ADSR-style amplitude envelope."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def generate(self, freq: float, duration: float, wave_type: str = 'saw',
                 amplitude: float = 0.3) -> np.ndarray:
        n = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n, False)

        if wave_type == 'sine':
            wave = np.sin(2 * np.pi * freq * t)
        elif wave_type == 'saw':
            wave = signal.sawtooth(2 * np.pi * freq * t)
        elif wave_type == 'square':
            wave = signal.square(2 * np.pi * freq * t, duty=0.5)
        else:
            raise ValueError("Unsupported wave_type.")

        env = np.ones(n)
        a = int(0.01 * self.sample_rate)
        d = int(0.1 * self.sample_rate)
        s_level = 0.7
        r = int(0.1 * self.sample_rate)

        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if d > 0 and (a + d) < n:
            env[a:a + d] = np.linspace(1, s_level, d)
            env[a + d:-r] = s_level
        if r > 0:
            env[-r:] = np.linspace(s_level, 0, r)

        return amplitude * wave * env


class ProsodyAuditor:
    """Map lyric stress patterns (via CMU dict) onto a beat grid."""

    def __init__(self) -> None:
        try:
            import nltk
            from nltk.corpus import cmudict
        except ImportError as exc:
            raise ImportError("ProsodyAuditor needs nltk installed") from exc
        try:
            nltk.data.find('corpora/cmudict')
        except LookupError:
            nltk.download('cmudict', quiet=True)
        self.dict = cmudict.dict()

    def get_stress_pattern(self, word: str, variant: int = 0) -> List[int]:
        word = word.lower().strip(".,!?\"'")
        if word not in self.dict:
            return [0]
        variants = self.dict[word]
        safe_variant = variant if variant < len(variants) else 0
        phonemes = variants[safe_variant]
        stresses = [int(char) for ph in phonemes for char in ph
                    if char.isdigit()]
        return [1 if s > 0 else 0 for s in stresses]

    def analyze_lyrics(self, lyrics: str) -> str:
        words = lyrics.split()
        beats = ['1', '&', '2', '&', '3', '&', '4', '&']
        analysis = []
        beat_idx = 0
        for word in words:
            stresses = self.get_stress_pattern(word)
            for stress in stresses:
                bar = (beat_idx // 8) + 1
                current_beat = beats[beat_idx % len(beats)]
                status = "STRONG" if stress == 1 else "weak"
                analysis.append(f"[Bar {bar} | Beat {current_beat}] "
                                f"{word} -> {status}")
                beat_idx += 1
        return "\n".join(analysis)


class VocalModel:
    """Vowel coloring via cascaded resonant (formant) filters."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def apply_vowel_formant(self, audio_signal: np.ndarray,
                            vowel: str = 'ah', bw: float = 80) -> np.ndarray:
        formants = {
            'ah': [730, 1090, 2440],
            'ee': [270, 2290, 3010],
            'oo': [300, 870, 2240],
        }
        if vowel not in formants:
            return audio_signal
        filtered_signal = np.zeros_like(audio_signal)
        for f in formants[vowel]:
            q_factor = f / bw
            b, a = signal.iirpeak(f, q_factor, fs=self.sample_rate)
            sos = signal.tf2sos(b, a)
            filtered_signal += signal.sosfilt(sos, audio_signal)
        return filtered_signal / len(formants[vowel])


class MixerBuss:
    """Sum stems, normalize to just under full scale, export 16-bit WAV."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate

    def sum_and_export(self, audio_arrays: List[np.ndarray],
                       filename: str = "output.wav") -> str:
        if not audio_arrays:
            raise ValueError("No audio arrays provided for mixing.")
        max_len = max(map(len, audio_arrays))
        padded = [np.pad(a, (0, max_len - len(a))) for a in audio_arrays]
        mixed = np.sum(padded, axis=0)
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = (mixed / peak) * 0.99
        mixed_16bit = (mixed * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, mixed_16bit)
        return (f"Successfully mixed and exported to {filename} "
                f"(Peak: {peak:.2f} normalized)")


def demo(output: str = "sovereign_ribcage.wav") -> str:
    """End-to-end demo: prosody, chord, synthesis, formants, mix, export."""
    harmony = HarmonyEngine()
    forge = SynthesisForge()
    vocal = VocalModel()
    mixer = MixerBuss()

    lyric = "Touch the stick you get the flame"
    try:
        prosody = ProsodyAuditor()
        print("--- 1. ANALYZING PROSODY ---")
        print(prosody.analyze_lyrics(lyric))
    except ImportError:
        print("--- 1. PROSODY SKIPPED (nltk not installed) ---")

    print("\n--- 2. CONSTRUCTING HARMONIC RIBCAGE ---")
    chord = harmony.generate_7th_chord('A', 'minor7', octave=3, drop_2=True)
    print(f"MIDI Notes (Drop-2): {chord['midi']}")
    print(f"Frequencies: {[round(f, 2) for f in chord['freqs']]}")

    print("\n--- 3. SYNTHESIS & MIXING ---")
    waves = [forge.generate(f, duration=3.0, wave_type='saw', amplitude=0.2)
             for f in chord['freqs']]
    vocalized = [vocal.apply_vowel_formant(w, 'ah') for w in waves]
    status = mixer.sum_and_export(vocalized, output)
    print(status)
    return status
