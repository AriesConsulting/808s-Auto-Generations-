import json
import wave

import pytest

from auto808 import minimax
from auto808.minimax import MiniMaxDatasetBuilder, PipelineConfig


def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.sample_rate == 32000
    assert cfg.channels == 2
    assert cfg.artist_name == "Aries Hilton"


def test_config_roundtrip():
    cfg = PipelineConfig(artist_name="Test", batch_size=7)
    d = cfg.to_dict()
    assert d["artist_name"] == "Test"
    cfg2 = PipelineConfig.from_dict({**d, "bogus_field": 1})
    assert cfg2.batch_size == 7


def test_sanitize_filename():
    s = MiniMaxDatasetBuilder._sanitize_filename("Hello World! (2024)")
    assert s == "hello_world_2024"
    assert MiniMaxDatasetBuilder._sanitize_filename("a") == "a"


def test_parse_catalog(tmp_path):
    cat = tmp_path / "catalog.txt"
    cat.write_text("# comment\n\nhttps://x/y | My Title | tag1\nhttps://x/z\n")
    cfg = PipelineConfig(catalog_file=str(cat),
                         workspace=str(tmp_path / "ws"))
    builder = MiniMaxDatasetBuilder(config=cfg)
    tracks = builder.parse_catalog()
    assert len(tracks) == 2
    assert tracks[0]["custom_title"] == "My Title"
    assert tracks[0]["tags"] == "tag1"
    assert tracks[1]["custom_title"] is None


def test_parse_catalog_missing(tmp_path):
    cfg = PipelineConfig(catalog_file=str(tmp_path / "nope.txt"),
                         workspace=str(tmp_path / "ws"))
    builder = MiniMaxDatasetBuilder(config=cfg)
    assert builder.parse_catalog() == []


def _write_wav(path, channels, rate, seconds=1):
    import numpy as np
    n = rate * seconds
    data = (np.zeros((n, channels)) * 32767).astype("int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def test_validate_audio_file(tmp_path):
    cfg = PipelineConfig(workspace=str(tmp_path / "ws"))
    builder = MiniMaxDatasetBuilder(config=cfg)
    good = tmp_path / "good.wav"
    _write_wav(good, 2, 32000)
    assert builder._validate_audio_file(good)
    bad_ch = tmp_path / "bad.wav"
    _write_wav(bad_ch, 1, 32000)
    assert not builder._validate_audio_file(bad_ch)
    tiny = tmp_path / "tiny.wav"
    tiny.write_bytes(b"RIFF")
    assert not builder._validate_audio_file(tiny)
    assert not builder._validate_audio_file(tmp_path / "missing.wav")


def test_workspace_and_checkpoint(tmp_path):
    cfg = PipelineConfig(workspace=str(tmp_path / "ws"))
    builder = MiniMaxDatasetBuilder(config=cfg)
    builder.setup_workspace()
    assert (tmp_path / "ws" / "dataset" / "audio").is_dir()
    assert (tmp_path / "ws" / "config" / "pipeline_config.yaml").exists()
    builder.save_checkpoint("0001_test")
    builder2 = MiniMaxDatasetBuilder(config=cfg)
    builder2.load_checkpoints()
    assert "0001_test" in builder2.processed_tracks


def test_lyrics_template_and_metadata(tmp_path):
    cfg = PipelineConfig(workspace=str(tmp_path / "ws"))
    builder = MiniMaxDatasetBuilder(config=cfg)
    builder.setup_workspace()
    lyric = builder.generate_lyrics_template("0001_x", "Some Title")
    text = lyric.read_text()
    assert "[CHORUS]" in text and "Some Title" in text
    wav_path = tmp_path / "a.wav"
    _write_wav(wav_path, 2, 32000)
    meta = builder.generate_metadata("0001_x", "Some Title", wav_path,
                                     tags="dark")
    assert meta["track_id"] == "0001_x"
    assert "Aries Hilton" in meta["caption"]
    assert "dark" in meta["caption"]


def test_cli_parser():
    p = minimax.build_parser()
    args = p.parse_args(["--workspace", "w", "--init-only"])
    assert args.workspace == "w" and args.init_only
