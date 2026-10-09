"""Video enhancement: delogo, 4K Lanczos upscale, 60fps interpolation.

Wraps ffmpeg/ffprobe. The binaries must be on PATH; no Python video
dependencies are needed. Extracted from the 4K UHD Upscaler notebook
(Colab upload/download steps removed; paths are plain arguments).
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict


def get_video_info(input_file: str) -> Dict[str, int]:
    """Return width/height of the first video stream via ffprobe."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height', '-of', 'json', input_file]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, check=True)
        info = json.loads(result.stdout)
        if not info.get('streams'):
            raise ValueError("Could not find video streams in the file.")
        stream = info['streams'][0]
        return {'width': int(stream['width']), 'height': int(stream['height'])}
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe failed: {e.stderr.strip()}")
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found on PATH")


def build_filter_chain(width: int, height: int) -> str:
    """delogo (bottom-right watermark) + 4K Lanczos scale + 60fps interpolate."""
    wm_w, wm_h = 200, 80
    wm_x = max(0, width - wm_w - 20)
    wm_y = max(0, height - wm_h - 20)
    delogo = f"delogo=x={wm_x}:y={wm_y}:w={wm_w}:h={wm_h}:show=0"
    scale = "scale=3840:-1:flags=lanczos"
    interpolate = ("minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:"
                   "me_mode=bidir:vsbmc=1")
    return f"{delogo},{scale},{interpolate}"


def default_output_name(input_file: str) -> str:
    base, _ext = os.path.splitext(input_file)
    return f"{base}_ENHANCED_4K_60FPS.mp4"


def process_video(input_file: str, output_file: str | None = None,
                  preset: str = 'slow', crf: str = '18') -> str:
    """Run the enhancement filter chain. Returns the output path."""
    if output_file is None:
        output_file = default_output_name(input_file)
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        raise ValueError("Input and output filenames cannot be the same.")

    meta = get_video_info(input_file)
    print(f"Input Resolution: {meta['width']}x{meta['height']}")
    print("Target Output: 4K (3840 wide) @ 60 FPS")

    cmd = ['ffmpeg', '-i', input_file,
           '-vf', build_filter_chain(meta['width'], meta['height']),
           '-c:v', 'libx264', '-preset', preset, '-crf', crf,
           '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-y', output_file]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH")
    print(f"Enhancement complete: {output_file}")
    return output_file
