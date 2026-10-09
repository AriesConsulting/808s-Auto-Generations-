import json
import os

import numpy as np
import pytest

from auto808 import vfx


def _ctx(tmp_path, frame=None):
    src = tmp_path / "shot.tif"
    src.write_bytes(b"fake-frame-bytes")
    return {"shot_name": "sc01_sh010", "input_path": str(src),
            "output_dir": str(tmp_path / "out"), "metadata": {}}


def test_pipeline_success_and_failure():
    ok_pipe = vfx.Pipeline("ok", [lambda c: {**c, "x": 1}])
    res = ok_pipe.run({"shot_name": "s"})
    assert res["status"] == "SUCCESS" and res["x"] == 1

    def boom(ctx):
        raise RuntimeError("nope")

    bad_pipe = vfx.Pipeline("bad", [boom])
    res = bad_pipe.run({"shot_name": "s"})
    assert res["status"] == "FAILED"
    assert "nope" in res["error"]


def test_ingest_hashes_input(tmp_path):
    ctx = vfx.ingest_exr_stage(_ctx(tmp_path))
    assert len(ctx["input_hash"]) == 64
    assert ctx["frame"].shape[2] == 3


def test_ingest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        vfx.ingest_exr_stage({"shot_name": "s",
                              "input_path": str(tmp_path / "nope.tif"),
                              "output_dir": "o", "metadata": {}})


def test_provenance_and_save(tmp_path):
    ctx = _ctx(tmp_path)
    ctx = vfx.ingest_exr_stage(ctx)
    ctx = vfx.ml_denoise_stage(ctx)
    ctx = vfx.ml_roto_stage(ctx)
    assert ctx["aovs"]["roto_matte"].shape[:2] == ctx["frame"].shape[:2]
    ctx = vfx.write_provenance_stage(ctx)
    assert ctx["provenance"]["shot"] == "sc01_sh010"
    ctx = vfx.save_exr_stage(ctx)
    assert os.path.exists(ctx["provenance_path"])
    vfx.qc_check_stage(ctx)  # raises on failure; silence means QC passed


def test_qc_rejects_nan(tmp_path):
    ctx = _ctx(tmp_path)
    ctx = vfx.ingest_exr_stage(ctx)
    ctx["frame"] = np.full((4, 4, 3), np.nan, dtype=np.float32)
    ctx["provenance_path"] = str(tmp_path / "p.json")
    (tmp_path / "p.json").write_text("{}")
    with pytest.raises(ValueError, match="NaN"):
        vfx.qc_check_stage(ctx)


def test_dispatcher_json_timeline(tmp_path):
    tl = tmp_path / "tl.json"
    tl.write_text(json.dumps({"clips": [
        {"name": "a", "source": "x"}, {"name": "b", "source": "y"}]}))
    farm = vfx.FarmDispatcher(vfx.default_pipeline())
    farm.parse_timeline(str(tl), output_dir="o")
    assert len(farm.job_queue) == 2
    assert farm.job_queue[0]["shot_name"] == "a"


def test_run_demo_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = vfx.run_demo(str(tmp_path))
    assert len(results) == 2
    assert all(r["status"] == "SUCCESS" for r in results)
    prov = json.loads(open(results[0]["provenance_path"]).read())
    assert prov["shot"] == "sc01_sh010"
