import pytest

from auto808 import upscale


def test_build_filter_chain():
    chain = upscale.build_filter_chain(1920, 1080)
    assert "delogo=x=1700:y=980:w=200:h=80" in chain
    assert "scale=3840:-1:flags=lanczos" in chain
    assert "minterpolate=fps=60" in chain


def test_default_output_name():
    assert upscale.default_output_name("clip.mov") == "clip_ENHANCED_4K_60FPS.mp4"
    assert upscale.default_output_name("/a/b/clip.avi") == \
        "/a/b/clip_ENHANCED_4K_60FPS.mp4"


def test_get_video_info_missing_binary_or_file(tmp_path):
    # Either ffprobe is absent or the file is: both are RuntimeError.
    with pytest.raises(RuntimeError):
        upscale.get_video_info(str(tmp_path / "nope.mp4"))


def test_process_video_same_in_out(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        upscale.process_video(str(src), str(src))
