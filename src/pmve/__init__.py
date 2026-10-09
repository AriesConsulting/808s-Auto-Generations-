"""pmve: Aries Hilton's music production and VFX pipeline toolkit.

Modules are importable independently so heavy optional dependencies
(opencv, librosa, torch-family) are only needed for the pieces you use:

- synth: harmony, waveform synthesis, vocal formants, stem mixing
- minimax: artist catalog to MiniMax Music 3 fine-tune dataset pipeline
- spatial: audio-reactive video distortion engines
- vfx: staged cinematic VFX pipeline with provenance ledger
- upscale: ffmpeg 4K/60fps video enhancement
- logo: Lucid Triangulation Records SVG logo generator
- vault: file integrity gate, Fernet vault, spectral audit, MIDI triggers
"""

__version__ = "0.1.0"
__author__ = "Aries Hilton"
