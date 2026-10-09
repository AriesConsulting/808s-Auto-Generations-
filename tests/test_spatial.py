import numpy as np
import pytest

from pmve import spatial


def test_missing_deps_raise_helpful_errors():
    try:
        import cv2  # noqa: F401
        cv2_missing = False
    except ImportError:
        cv2_missing = True
    try:
        import librosa  # noqa: F401
        librosa_missing = False
    except ImportError:
        librosa_missing = True

    if cv2_missing:
        with pytest.raises(ImportError, match="opencv-python"):
            spatial._need_cv2()
    if librosa_missing:
        with pytest.raises(ImportError, match="librosa"):
            spatial._need_librosa()
            spatial.analyze_audio("/nope.wav")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None,
    reason="opencv not installed")
def test_v1_distortion_shape():
    eng = spatial.SpatialQuantumEngineV1()
    frame = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    out = eng.apply_unified_distortion(frame, 32, 32, 0.5)
    assert out.shape == (64, 64, 3)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None,
    reason="opencv not installed")
def test_v2_process_frame_shape():
    eng = spatial.SpatialQuantumEngine()
    frame = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    audio = {"bass": 0.5, "mids": 0.5, "highs": 0.5}
    out = eng.process_frame(frame, audio, 0.0, 0)
    assert out.shape == (64, 128, 3)  # side-by-side with dot grid
