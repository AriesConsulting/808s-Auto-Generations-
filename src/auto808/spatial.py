"""Audio-reactive video distortion engines.

Two generations of the same idea: warp video frames with a gravitational
pull field driven by audio loudness/frequency bands, add chromatic
aberration, motion ghosting, and a dot grid. V2 (3D stems) splits the
audio into bass/mid/high bands via STFT and drives each visual element
from its own band.

Needs: opencv-python, numpy, librosa. The render functions take explicit
file paths (the notebooks used a tkinter file picker).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _need_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise ImportError("spatial needs opencv-python installed") from exc


def _need_librosa():
    try:
        import librosa
        return librosa
    except ImportError as exc:
        raise ImportError("spatial needs librosa installed") from exc


class SpatialQuantumEngineV1:
    """First generation: single loudness value drives pull and warp."""

    def __init__(self) -> None:
        self.ghost_layer = None
        self.velocity_field = None
        self.decay_rate = 0.92

    def apply_unified_distortion(self, img: np.ndarray, sun_x: float,
                                 sun_y: float, loudness: float,
                                 chromatic_strength: float = 0.22) -> np.ndarray:
        cv2 = _need_cv2()
        h, w = img.shape[:2]
        if self.velocity_field is None:
            self.velocity_field = np.zeros((h, w, 2), dtype=np.float32)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx, dy = x - sun_x, y - sun_y
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
        pull_mag = (loudness * 85) / (dist / 45 + 1)
        current_force_x = (dx / dist) * pull_mag
        current_force_y = (dy / dist) * pull_mag

        self.velocity_field[:, :, 0] = (
            self.velocity_field[:, :, 0] + current_force_x * 0.2) * self.decay_rate
        self.velocity_field[:, :, 1] = (
            self.velocity_field[:, :, 1] + current_force_y * 0.2) * self.decay_rate

        flex_x = x - self.velocity_field[:, :, 0]
        flex_y = y - self.velocity_field[:, :, 1]

        x_n = (2.0 * flex_x - w) / w
        y_n = (2.0 * flex_y - h) / h
        r_radial = np.sqrt(x_n ** 2 + y_n ** 2)

        output = []
        for s_mult in (1.08, 1.0, 0.92):
            s = chromatic_strength * s_mult
            dist_factor = 1.0 + s * (r_radial ** 2)
            mx = (((x_n * dist_factor) + 1.0) * w / 2.0).astype(np.float32)
            my = (((y_n * dist_factor) + 1.0) * h / 2.0).astype(np.float32)
            output.append(cv2.remap(img, mx, my, cv2.INTER_LINEAR))

        return cv2.merge([output[0][:, :, 0],
                          output[1][:, :, 1],
                          output[2][:, :, 2]])

    def process_frame(self, frame: np.ndarray, loudness: float,
                      pan_x: float, pan_y: float,
                      freq_color: tuple) -> np.ndarray:
        cv2 = _need_cv2()
        h, w, _ = frame.shape
        if self.ghost_layer is None:
            self.ghost_layer = np.zeros_like(frame)

        target_x = int((w / 2) + (pan_x * w / 3))
        target_y = int((h / 2) - (pan_y * h / 3))

        sun_radius = int(25 + (loudness * 80))
        sun_overlay = np.zeros_like(frame)
        cv2.circle(sun_overlay, (target_x, target_y), sun_radius,
                   freq_color, -1)
        sun_overlay = cv2.GaussianBlur(sun_overlay, (75, 75), 0)

        self.ghost_layer = cv2.addWeighted(self.ghost_layer, 0.88,
                                            sun_overlay, 0.20, 0)

        b, g, r = cv2.split(frame)
        shift_amount = int(loudness * 30)
        r = np.roll(r, shift_amount, axis=1)
        b = np.roll(b, -shift_amount, axis=1)
        glitched = cv2.merge([b, g, r])

        warped_base = self.apply_unified_distortion(glitched, target_x,
                                                    target_y, loudness)
        combined_visual = cv2.add(warped_base, self.ghost_layer)

        dot_canvas = np.zeros_like(frame)
        dot_res = 24
        for y_step in range(0, h, dot_res):
            for x_step in range(0, w, dot_res):
                d_to_sun = np.sqrt((x_step - target_x) ** 2 +
                                   (y_step - target_y) ** 2)
                dot_r = max(1, int(14 * (1 - d_to_sun / w) * loudness))
                cv2.circle(dot_canvas, (x_step, y_step), dot_r,
                           freq_color, -1)

        return np.hstack((combined_visual, dot_canvas))


class SpatialQuantumEngine:
    """Second generation (3D stems): bass/mid/high bands drive the visuals."""

    def __init__(self) -> None:
        self.ghost_layer = None
        self.velocity_field = None
        self.decay_rate = 0.90
        self.color_phase = 0

    def apply_quantum_physics(self, img: np.ndarray, sun_x: float,
                              sun_y: float, bass_force: float,
                              mid_force: float,
                              chromatic_strength: float = 0.25) -> np.ndarray:
        cv2 = _need_cv2()
        h, w = img.shape[:2]
        if self.velocity_field is None:
            self.velocity_field = np.zeros((h, w, 2), dtype=np.float32)

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        dx, dy = x - sun_x, y - sun_y
        dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6

        pull_mag = (bass_force * 110) / (dist / 40 + 1)
        current_force_x = (dx / dist) * pull_mag
        current_force_y = (dy / dist) * pull_mag

        self.velocity_field[:, :, 0] = (
            self.velocity_field[:, :, 0] + current_force_x * 0.3) * self.decay_rate
        self.velocity_field[:, :, 1] = (
            self.velocity_field[:, :, 1] + current_force_y * 0.3) * self.decay_rate

        flex_x = x - self.velocity_field[:, :, 0]
        flex_y = y - self.velocity_field[:, :, 1]

        x_n = (2.0 * flex_x - w) / w
        y_n = (2.0 * flex_y - h) / h
        r_radial = np.sqrt(x_n ** 2 + y_n ** 2)

        output_channels = []
        for i, s_mult in enumerate((1.12, 1.0, 0.88)):
            s = chromatic_strength * s_mult * (1 + mid_force)
            dist_factor = 1.0 + s * (r_radial ** 2)
            mx = (((x_n * dist_factor) + 1.0) * w / 2.0).astype(np.float32)
            my = (((y_n * dist_factor) + 1.0) * h / 2.0).astype(np.float32)
            ch_remap = cv2.remap(img[:, :, i], mx, my, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            output_channels.append(ch_remap)

        return cv2.merge(output_channels)

    def process_frame(self, frame: np.ndarray, audio_data: Dict[str, float],
                      pan_x: float, f_idx: int) -> np.ndarray:
        cv2 = _need_cv2()
        h, w, _ = frame.shape
        if self.ghost_layer is None:
            self.ghost_layer = np.zeros_like(frame)

        bass, mids, highs = (audio_data['bass'], audio_data['mids'],
                             audio_data['highs'])

        target_x = int((w / 2) + (pan_x * w / 2.5))
        target_y = int((h / 2) + (math.sin(f_idx * 0.1) * h / 10) * mids)

        sun_radius = int(20 + (bass * 60) + (mids * 40))
        sun_color = (int(127 + 127 * math.sin(f_idx * 0.05)),
                     int(127 + 127 * math.cos(f_idx * 0.03)), 255)
        sun_overlay = np.zeros_like(frame)
        cv2.circle(sun_overlay, (target_x, target_y), sun_radius,
                   sun_color, -1)
        blur_amt = int(41 + (30 * mids))
        if blur_amt % 2 == 0:
            blur_amt += 1
        sun_overlay = cv2.GaussianBlur(sun_overlay, (blur_amt, blur_amt), 0)

        self.ghost_layer = cv2.addWeighted(self.ghost_layer, 0.85,
                                            sun_overlay, 0.3, 0)
        warped_base = self.apply_quantum_physics(frame, target_x, target_y,
                                                 bass, mids)
        combined_visual = cv2.add(warped_base, self.ghost_layer)

        dot_canvas = np.zeros_like(frame)
        res = 20
        for y_step in range(0, h, res):
            for x_step in range(0, w, res):
                d = np.sqrt((x_step - target_x) ** 2 +
                            (y_step - target_y) ** 2)
                jitter = int((np.random.rand() - 0.5) * 10 * highs)
                dot_r = max(1, int(10 * (1 - d / w) * highs + (2 * bass)))
                cv2.circle(dot_canvas, (x_step + jitter, y_step + jitter),
                           dot_r, sun_color, -1)

        return np.hstack((combined_visual, dot_canvas))


def analyze_audio(path: str) -> Dict[str, np.ndarray]:
    """Split audio into normalized bass/mid/high bands plus stereo pan."""
    librosa = _need_librosa()
    y, sr = librosa.load(path, mono=False)
    if y.ndim > 1:
        left, right = y[0], y[1]
        y_mono = librosa.to_mono(y)
        rms_l = librosa.feature.rms(y=left)[0]
        rms_r = librosa.feature.rms(y=right)[0]
        pan = (rms_r - rms_l) / (rms_l + rms_r + 1e-6)
    else:
        y_mono = y
        pan = np.zeros(100)

    stft = np.abs(librosa.stft(y_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_band = stft[(freqs >= 20) & (freqs <= 150)].mean(axis=0)
    mid_band = stft[(freqs > 150) & (freqs <= 2500)].mean(axis=0)
    high_band = stft[(freqs > 2500)].mean(axis=0)

    def norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    return {'bass': norm(bass_band), 'mids': norm(mid_band),
            'highs': norm(high_band), 'pan': pan,
            'time_stamps': len(bass_band)}


def render(input_path: str, output_path: str = "quantum_fft_output.mp4",
           engine: str = "v2", preview: bool = False) -> str:
    """Render the audio-reactive video. engine is 'v1' or 'v2'."""
    cv2 = _need_cv2()
    librosa = _need_librosa()

    y, _sr = librosa.load(input_path, mono=False)
    if engine == "v1":
        eng: object = SpatialQuantumEngineV1()
        if y.ndim > 1:
            rms_l = librosa.feature.rms(y=y[0])[0]
            rms_r = librosa.feature.rms(y=y[1])[0]
            pan = (rms_r - rms_l) / (rms_l + rms_r + 1e-6)
            loud = (rms_l + rms_r) / 2
        else:
            pan = np.zeros(100)
            loud = librosa.feature.rms(y=y)[0]
        loud = ((loud - loud.min()) /
                (loud.max() - loud.min() + 1e-6))
        is_stereo = y.ndim > 1
        audio = None
    else:
        eng = SpatialQuantumEngine()
        audio = analyze_audio(input_path)
        loud, pan, is_stereo = None, audio['pan'], isinstance(
            audio['pan'], np.ndarray)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    steps = audio['time_stamps'] if audio else len(loud)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w * 2, h))
    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        a_idx = min(int((f_idx / total_f) * steps), steps - 1)
        if engine == "v1":
            cur_loud = float(loud[a_idx])
            cur_pan = float(pan[a_idx]) if is_stereo else 0.0
            cur_pan_y = (np.sin(f_idx * 0.08) * 0.15) * cur_loud
            color = (int(127 + 127 * np.sin(f_idx * 0.04)),
                     int(127 + 127 * np.cos(f_idx * 0.02)), 200)
            res = eng.process_frame(frame, cur_loud, cur_pan,
                                    cur_pan_y, color)
        else:
            cur = {'bass': float(audio['bass'][a_idx]),
                   'mids': float(audio['mids'][a_idx]),
                   'highs': float(audio['highs'][a_idx])}
            cur_pan = float(pan[a_idx]) if is_stereo else 0.0
            res = eng.process_frame(frame, cur, cur_pan, f_idx)
        out.write(res)
        if preview:
            cv2.imshow('Quantum Engine', cv2.resize(res, (1200, 500)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        f_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Rendered {f_idx} frames to {output_path}")
    return output_path
