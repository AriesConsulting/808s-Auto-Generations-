#!/usr/bin/env python3
"""pmve command line: every pipeline from the notebooks, one entry point.

Usage:
    python cli.py synth-demo [--out out.wav]
    python cli.py minimax [--workspace DIR] [--catalog FILE] [--init-only]
    python cli.py spatial INPUT [--engine v1|v2] [--out out.mp4]
    python cli.py vfx [--dir DIR]
    python cli.py upscale INPUT [--out out.mp4]
    python cli.py logo [--out logo.svg]
    python cli.py vault-demo
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "src"))

from pmve import __version__  # noqa: E402


def cmd_synth_demo(args: argparse.Namespace) -> int:
    from pmve import synth
    synth.demo(args.out)
    return 0


def cmd_minimax(args: argparse.Namespace) -> int:
    from pmve import minimax
    argv = ["--workspace", args.workspace, "--catalog", args.catalog]
    if args.config:
        argv += ["--config", args.config]
    if args.init_only:
        argv.append("--init-only")
    return minimax.main(argv)


def cmd_spatial(args: argparse.Namespace) -> int:
    from pmve import spatial
    try:
        spatial.render(args.input, output_path=args.out,
                       engine=args.engine, preview=args.preview)
    except ImportError as e:
        print(f"cannot run: {e}")
        return 1
    return 0


def cmd_vfx(args: argparse.Namespace) -> int:
    from pmve import vfx
    vfx.run_demo(args.dir)
    return 0


def cmd_upscale(args: argparse.Namespace) -> int:
    from pmve import upscale
    try:
        upscale.process_video(args.input, args.out)
    except RuntimeError as e:
        print(f"cannot run: {e}")
        return 1
    return 0


def cmd_logo(args: argparse.Namespace) -> int:
    from pmve import logo
    try:
        logo.create_lucid_triangulation_logo(args.out)
    except ImportError as e:
        print(f"cannot run: {e}")
        return 1
    return 0


def cmd_vault_demo(args: argparse.Namespace) -> int:
    from pmve import vault
    vault.demo()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="pmve",
        description=f"pmve {__version__}: music production and VFX pipelines")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth-demo", help="synthesize a chord and mix to WAV")
    s.add_argument("--out", default="sovereign_ribcage.wav")
    s.set_defaults(func=cmd_synth_demo)

    m = sub.add_parser("minimax", help="MiniMax Music 3 dataset builder")
    m.add_argument("--workspace", default="minimax_artist_workspace")
    m.add_argument("--catalog", default="catalog/youtube_catalog.txt")
    m.add_argument("--config", default=None)
    m.add_argument("--init-only", action="store_true")
    m.set_defaults(func=cmd_minimax)

    p = sub.add_parser("spatial", help="audio-reactive video render")
    p.add_argument("input", help="video file with an audio track")
    p.add_argument("--engine", choices=["v1", "v2"], default="v2")
    p.add_argument("--out", default="quantum_fft_output.mp4")
    p.add_argument("--preview", action="store_true")
    p.set_defaults(func=cmd_spatial)

    v = sub.add_parser("vfx", help="cinematic VFX pipeline smoke run")
    v.add_argument("--dir", default=".")
    v.set_defaults(func=cmd_vfx)

    u = sub.add_parser("upscale", help="4K/60fps ffmpeg video enhancement")
    u.add_argument("input")
    u.add_argument("--out", default=None)
    u.set_defaults(func=cmd_upscale)

    lg = sub.add_parser("logo", help="render the label logo SVG")
    lg.add_argument("--out", default="lucid_triangulation_logo.svg")
    lg.set_defaults(func=cmd_logo)

    vd = sub.add_parser("vault-demo", help="integrity vault demo run")
    vd.set_defaults(func=cmd_vault_demo)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
