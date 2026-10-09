# auto808

Music production and VFX pipelines by Aries Hilton / Frozen In Flames.
Rebuilt from the original Colab notebooks into a proper Python package:
one importable module per pipeline, a single CLI, and a test suite.

Apache-2.0. The original notebooks are preserved in `notebooks/` for
provenance.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Core modules need only `numpy`, `scipy`, `pandas`, `pyyaml`,
`cryptography`, `svgwrite`, and `mido`. Heavier pieces are optional and
degrade gracefully when missing: `librosa` + `opencv-python` for the
spatial engine, `imageio` / `onnxruntime` / `opentimelineio` for VFX
backends, `nltk` for lyric prosody.

## CLI

```bash
python cli.py synth-demo [--out out.wav]
python cli.py minimax [--workspace DIR] [--catalog catalog.txt] [--init-only]
python cli.py spatial INPUT.mp4 [--engine v1|v2] [--out out.mp4]
python cli.py vfx [--dir DIR]
python cli.py upscale INPUT.mp4 [--out out.mp4]
python cli.py logo [--out logo.svg]
python cli.py vault-demo
```

## Modules

### synth (Sovereign Production Engine)
DSP toolkit for building tracks from scratch.
`HarmonyEngine` builds 7th chords (major7, minor7, dominant7, m7b5, with
drop-2 voicings) and converts note names to MIDI to frequencies.
`SynthesisForge` renders sine/saw/square oscillators through an
ADSR-style envelope. `VocalModel` colors audio with vowel formant
filters (ah, ee, oo). `MixerBuss` sums stems, normalizes under full
scale, and exports 16-bit WAV. `ProsodyAuditor` maps lyric stress
patterns from the CMU pronouncing dictionary onto a beat grid
(needs nltk).

### minimax (MiniMax Music 3 dataset pipeline)
Turns an artist catalog into a fine-tune dataset for MiniMax Music 3:
32kHz stereo WAV extraction via yt-dlp, loudness normalization,
structured lyric templates with section markers, librosa feature
extraction (BPM, key, energy, mood), and CSV/JSON metadata.
Checkpoint/resume so interrupted runs pick up where they left off.
Catalog format: one `url | title | tags` per line.

### spatial (Spatial Quantum Engine, v1 + v2)
Audio-reactive video distortion. A gravitational pull field driven by
loudness warps each frame, with chromatic aberration, motion ghosting,
and a dot grid layered on top. V2 splits audio into bass/mid/high
bands via STFT and drives each visual element from its own band
(bass pulls, mids bend the lens, highs flicker the grid).

### vfx (Cinematic VFX Pipeline Orchestrator)
A staged frame pipeline (ingest, denoise, roto matte, provenance,
save, QC) plus a `FarmDispatcher` that turns an editorial timeline
into a job queue. Every stage records into a cryptographic provenance
ledger (input SHA-256, models used, review flags). ONNX models,
imageio, and OpenTimelineIO are optional; without them each stage
runs a deterministic stub so the pipeline still executes end to end.

### upscale (4K UHD Video Enhancer)
ffmpeg wrapper: removes a bottom-right watermark (delogo), upscales
to 4K with Lanczos, and interpolates to 60fps. Needs the `ffmpeg`
and `ffprobe` binaries on PATH.

### logo (Lucid Triangulation Records logo)
Generates the label crest as SVG: three offset triangles (red, blue,
purple), a white core, and the wordmark in spaced capitals.

### vault (LTR integrity gate)
File integrity and encryption utilities: SHA-256 hash a file and gate
on the digest, encrypt labor blobs with Fernet, run a spectral audit
(Butterworth low-pass plus variance check) on audio buffers, and
detect a configurable MIDI trigger note sequence.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Heavy-dependency tests skip cleanly when the dependency is absent.
