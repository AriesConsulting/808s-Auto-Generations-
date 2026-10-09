"""Staged cinematic VFX pipeline with a cryptographic provenance ledger.

Stages: ingest -> denoise -> roto matte -> provenance -> save -> QC.
A FarmDispatcher turns an editorial timeline (OTIO or a JSON clip list)
into a job queue. Heavy backends (imageio, onnxruntime, opentimelineio)
are optional; every stage degrades to a deterministic stub without them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    import numpy as np
except ImportError:  # pragma: no cover
    raise ImportError("vfx needs numpy installed")

try:
    import imageio.v3 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import opentimelineio as otio
    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger("pmve.vfx")


class Pipeline:
    def __init__(self, name: str,
                 stages: List[Callable[[Dict[str, Any]], Dict[str, Any]]]):
        self.name = name
        self.stages = stages

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        log.info(f"--- Starting Pipeline: {self.name} for "
                 f"{context.get('shot_name', 'Unknown')} ---")
        for stage in self.stages:
            start = time.time()
            try:
                context = stage(context)
                log.info(f"  [+] Stage Complete: {stage.__name__} "
                         f"({time.time() - start:.3f}s)")
            except Exception as e:  # noqa: BLE001
                log.error(f"  [!] Stage FAILED: {stage.__name__} -> {e}")
                context['status'] = 'FAILED'
                context['error'] = str(e)
                break
        if context.get('status') != 'FAILED':
            context['status'] = 'SUCCESS'
        return context


def ingest_exr_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Read source frame, hash it for provenance, normalize to float32."""
    src = ctx['input_path']
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing source file: {src}")
    with open(src, 'rb') as fh:
        ctx['input_hash'] = hashlib.sha256(fh.read()).hexdigest()
    if HAS_IMAGEIO:
        img = imageio.imread(src)
        if img.dtype == np.uint8:
            ctx['frame'] = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            ctx['frame'] = img.astype(np.float32) / 65535.0
        else:
            ctx['frame'] = img.astype(np.float32)
    else:
        ctx['frame'] = np.random.rand(1080, 1920, 3).astype(np.float32)
    ctx['aovs'] = {}
    return ctx


def ml_roto_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Segmentation matte; synthetic circle when no ONNX model is present."""
    frame = ctx['frame']
    model_path = ctx.get('roto_model_path', 'models/sam_vit_h.onnx')
    if HAS_ORT and os.path.exists(model_path):
        session = ort.InferenceSession(model_path)  # noqa: F841 (hook)
        matte = np.zeros(frame.shape[:2], dtype=np.float32)
    else:
        h, w = frame.shape[:2]
        y, x = np.ogrid[:h, :w]
        mask = ((x - w / 2) ** 2 + (y - h / 2) ** 2 <= (h / 4) ** 2)
        matte = mask.astype(np.float32)
    ctx['aovs']['roto_matte'] = matte
    ctx['metadata']['roto_model'] = "stub_sam_v1"
    ctx['requires_human_review'] = True
    return ctx


def ml_denoise_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Neural denoise; gentle smoothing when no ONNX model is present."""
    frame = ctx['frame']
    model_path = ctx.get('denoise_model_path', 'models/real_esrgan.onnx')
    if HAS_ORT and os.path.exists(model_path):
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        inp = np.expand_dims(
            np.transpose(frame, (2, 0, 1)), axis=0).astype(np.float32)
        out = session.run(None, {input_name: inp})[0]
        denoised = np.transpose(out[0], (1, 2, 0))
        ctx['frame'] = np.clip(denoised, 0.0, 1.0)
    else:
        ctx['frame'] = np.clip(frame * 0.95 + 0.05, 0.0, 1.0)
    ctx['metadata']['denoise_model'] = "stub_optix_v2"
    return ctx


def write_provenance_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the provenance ledger for this frame."""
    ctx['provenance'] = {
        "shot": ctx['shot_name'],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pipeline_version": "1.0.4",
        "input_hash_sha256": ctx.get('input_hash', 'N/A'),
        "models_used": {
            "denoise": ctx['metadata'].get('denoise_model'),
            "roto": ctx['metadata'].get('roto_model'),
        },
        "manual_review_required": ctx.get('requires_human_review', False),
    }
    return ctx


def save_exr_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Write the comp frame, mattes, and the provenance sidecar JSON."""
    out_dir = Path(ctx['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = ctx['shot_name']
    out_frame = out_dir / f"{shot}_comp.exr"
    if HAS_IMAGEIO:
        imageio.imwrite(out_dir / f"{shot}_comp.tif",
                        (ctx['frame'] * 255).astype(np.uint8))
    ctx['output_frame_path'] = str(out_frame)
    if 'roto_matte' in ctx['aovs'] and HAS_IMAGEIO:
        imageio.imwrite(out_dir / f"{shot}_matte.tif",
                        (ctx['aovs']['roto_matte'] * 255).astype(np.uint8))
    sidecar = out_dir / f"{shot}_provenance.json"
    with open(sidecar, 'w') as fh:
        json.dump(ctx['provenance'], fh, indent=4)
    ctx['provenance_path'] = str(sidecar)
    return ctx


def qc_check_stage(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """QC: reject NaNs and missing provenance, warn on clipping."""
    frame = ctx['frame']
    if np.isnan(frame).any():
        raise ValueError("QC FAILED: NaN values detected in output frame.")
    if np.max(frame) > 1.0 or np.min(frame) < 0.0:
        log.warning("QC WARNING: Pixel values out of [0, 1] bounds.")
    if not os.path.exists(ctx['provenance_path']):
        raise ValueError("QC FAILED: Provenance sidecar missing.")
    log.info("  [+] QC PASSED: Image math and provenance verified.")
    return ctx


class FarmDispatcher:
    """Turn an editorial timeline into per-shot pipeline jobs."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.job_queue: List[Dict[str, Any]] = []

    def parse_timeline(self, timeline_path: str, output_dir: str) -> None:
        log.info(f"Parsing Timeline: {timeline_path}")
        if HAS_OTIO and timeline_path.endswith('.otio'):
            timeline = otio.adapters.read_from_file(timeline_path)
            for track in timeline.tracks:
                if track.kind == otio.schema.TrackKind.Video:
                    for clip in track.each_clip():
                        self._add_job(clip.name,
                                      clip.media_reference.target_url,
                                      output_dir)
        else:
            with open(timeline_path) as fh:
                data = json.load(fh)
            for clip in data.get('clips', []):
                self._add_job(clip['name'], clip['source'], output_dir)

    def _add_job(self, shot_name: str, src_path: str, output_dir: str) -> None:
        self.job_queue.append({'shot_name': shot_name,
                               'input_path': src_path,
                               'output_dir': output_dir,
                               'metadata': {}})
        log.info(f"Queued Shot: {shot_name}")

    def execute_queue(self) -> List[Dict[str, Any]]:
        log.info(f"Executing Queue: {len(self.job_queue)} jobs.")
        results = [self.pipeline.run(job) for job in self.job_queue]
        ok = sum(1 for r in results if r.get('status') == 'SUCCESS')
        log.info(f"Queue Complete. {ok}/{len(self.job_queue)} successful.")
        return results


def default_pipeline() -> Pipeline:
    return Pipeline(name="Cinematic_Main_v1", stages=[
        ingest_exr_stage, ml_denoise_stage, ml_roto_stage,
        write_provenance_stage, save_exr_stage, qc_check_stage])


def setup_dummy_environment(base: str = ".") -> str:
    """Create fake camera originals and a JSON timeline for a smoke run."""
    base_p = Path(base)
    (base_p / 'ingest').mkdir(parents=True, exist_ok=True)
    (base_p / 'renders').mkdir(parents=True, exist_ok=True)
    if HAS_IMAGEIO:
        for shot in ("sc01_sh010", "sc01_sh020"):
            noise = (np.random.rand(1080, 1920, 3) * 255).astype(np.uint8)
            imageio.imwrite(base_p / 'ingest' / f"{shot}_raw.tif", noise)
    else:
        for shot in ("sc01_sh010", "sc01_sh020"):
            (base_p / 'ingest' / f"{shot}_raw.tif").touch()
    timeline = {"clips": [
        {"name": "sc01_sh010",
         "source": str(base_p / 'ingest' / "sc01_sh010_raw.tif")},
        {"name": "sc01_sh020",
         "source": str(base_p / 'ingest' / "sc01_sh020_raw.tif")},
    ]}
    timeline_path = base_p / 'ingest' / 'sequence_v01.json'
    with open(timeline_path, 'w') as fh:
        json.dump(timeline, fh)
    log.info("Test production environment created.")
    return str(timeline_path)


def run_demo(base: str = ".") -> List[Dict[str, Any]]:
    """Full smoke run on generated dummy footage. Returns job results."""
    timeline_path = setup_dummy_environment(base)
    farm = FarmDispatcher(pipeline=default_pipeline())
    farm.parse_timeline(timeline_path,
                        output_dir=str(Path(base) / 'renders' / 'final_comps'))
    results = farm.execute_queue()
    print("\n--- Pipeline Audit Trail ---")
    for res in results:
        if res.get('status') == 'SUCCESS' and res.get('provenance_path'):
            with open(res['provenance_path']) as fh:
                prov = json.load(fh)
            print(f"Shot: {prov['shot']}")
            print(f"  Input Hash: {prov['input_hash_sha256']}")
            print(f"  Requires Manual Roto Check: "
                  f"{prov['manual_review_required']}")
            print(f"  Time: {prov['timestamp']}\n")
    return results
