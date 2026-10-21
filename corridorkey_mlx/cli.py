# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Command-line inference for CorridorKey v2 MLX profiles."""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import tempfile
import time
from contextlib import nullcontext
import statistics


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def strength(value):
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="corridorkey-v2-mlx 0.1.0")
    p.add_argument("--input", type=Path, help="Still image, video file, or directory of ordered image frames")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path)
    p.add_argument("--model-root", "--model_root", dest="model_root", type=Path,
                   default=Path("models"), help="Directory containing HxW-tN profiles")
    p.add_argument("--precision", choices=["fp32", "fp16", "q8"], default="fp32",
                   help="FP32 default; FP16/Q8 load the matching subfolder in the model repository")
    p.add_argument("--height", type=positive_int, default=1024, help="Padded profile height")
    p.add_argument("--width", type=positive_int, default=1536, help="Padded profile width")
    p.add_argument("--hann-chunk", "--hann_chunk", dest="hann_chunk", type=positive_int, default=4)
    p.add_argument("--hann-stride", "--hann_stride", dest="hann_stride", type=positive_int, default=2)
    p.add_argument("--despill-strength", "--despill_strength", dest="despill_strength", type=strength, default=1.)
    p.add_argument("--image-sequence", "--image_sequence", dest="image_sequence", action="store_true",
                   help="Write lossless 16-bit PNGs instead of video")
    p.add_argument("--low-vram", "--low_vram", dest="low_vram", action="store_true")
    p.add_argument("--compile", action="store_true", help="Compile MLX graphs for this shape")
    p.add_argument("--kernel-mode", choices=["enabled", "reference"], default="enabled",
                   help="Enabled optimizations by default; reference disables custom graph rewrites/kernels")
    p.add_argument("--benchmark-json", type=Path, help="Write measured CLI workflow timings and physical footprint")
    p.add_argument("--benchmark-runs", type=positive_int, default=1, help="Measured runs, reusing the loaded model")
    p.add_argument("--warmup-runs", type=nonnegative_int, default=0, help="Untimed warm-ups before measured runs")
    p.add_argument("--outputs", nargs="+", choices=["alpha", "fg", "raw_fg", "checker", "cutout"],
                   default=["alpha", "fg", "checker", "cutout"])
    p.add_argument("--hint-first-frame", "--hint_first_frame", dest="hint_first_frame", type=Path)
    p.add_argument("--hint-video", "--hint_video", dest="hint_video", type=Path)
    p.add_argument("--threads", type=positive_int, default=8, help="CPU I/O thread count")
    p.add_argument("--screen", choices=["green", "blue"], default="green", help="Despill channel")
    p.add_argument("--list-profiles", action="store_true", help="List local profiles without loading weights")
    p.add_argument("--check", action="store_true", help="Check input and required profile files without inference")
    return p


def required_files(root, height, width, frames):
    one = root / f"{height}x{width}-t1"
    result = [one / "stats.json"]
    for component in ("rvm", "foundations", "stage1", "stage2"):
        result.extend(one / component / name for name in ("graph.json", "weights.safetensors"))
    if frames != 1:
        result.extend(root / f"{height}x{width}-t{frames}" / "stage1" / name
                      for name in ("graph.json", "weights.safetensors"))
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.precision != "fp32":
        args.model_root = args.model_root / args.precision
    if args.list_profiles:
        profiles = []
        for profile in sorted(args.model_root.glob("*x*-t*")):
            if profile.is_dir():
                profiles.append({"profile": profile.name, "components": sorted(
                    x.name for x in profile.iterdir()
                    if x.is_dir() and (x / "graph.json").is_file()
                    and (x / "weights.safetensors").is_file())})
        print(json.dumps(profiles, indent=2))
        return 0
    if args.input is None or (args.output_dir is None and not args.check):
        p.error("--input and --output-dir are required for inference; --check needs only --input")
    if args.height % 128 or args.width % 128:
        p.error("profile height and width must be multiples of 128")
    if args.hann_stride > args.hann_chunk:
        p.error("--hann-stride cannot exceed --hann-chunk")
    if not args.input.exists():
        p.error(f"input does not exist: {args.input}")
    for hint in (args.hint_first_frame, args.hint_video):
        if hint is not None and not hint.exists():
            p.error(f"hint does not exist: {hint}")
    if args.output_dir is not None and args.input.is_dir():
        source, target = args.input.resolve(), args.output_dir.resolve()
        if source == target or source.is_relative_to(target) or target.is_relative_to(source):
            p.error("input and output directories must be separate, without nesting")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        p.error("inference requires Apple Silicon macOS")
    try:
        import torch
        from .pipeline import Source, run
        source = Source(args.input)
        if source.padded != (args.height, args.width):
            h, w = source.padded
            p.error(f"input pads to {h}x{w}; select --height {h} --width {w} and matching profiles")
        frames = min(args.hann_chunk, source.count)
        if frames < 1:
            p.error("input contains no frames")
        missing = [str(x) for x in required_files(args.model_root, args.height, args.width, frames)
                   if not x.is_file()]
        if missing:
            p.error("missing profile files:\n  " + "\n  ".join(missing)
                    + "\nDownload the matching authorized profiles or export them; see docs/MODELS.md.")
        expected_precision = {"fp32": "fp32", "fp16": "fp16-mixed-v1", "q8": "q8-fp16-v1"}[args.precision]
        for path in required_files(args.model_root, args.height, args.width, frames):
            if path.name == "graph.json":
                declared = json.loads(path.read_text()).get("precision", "fp32")
                if declared != expected_precision:
                    p.error(f"profile {path} declares {declared}, inconsistent with --precision {args.precision}")
        if not args.image_sequence:
            import shutil
            if shutil.which("ffmpeg") is None:
                p.error("Install ffmpeg for video export or use --image-sequence")
        if args.check:
            print(json.dumps({"status": "ready", "shape_hw": list(source.padded),
                              "frames": source.count, "stage1_window": frames}, indent=2))
            return 0
        from .mlx_backend import MLXModel
        torch.set_num_threads(args.threads)
        load_start = time.perf_counter()
        model = MLXModel(args.model_root, args.height, args.width, args.compile,
                         optimizations=() if args.kernel_mode == "reference" else None)
        model_load_seconds = time.perf_counter() - load_start
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {"status": "running", "precision": args.precision, "kernel_mode": args.kernel_mode,
                  "input": args.input.name, "frames": source.count, "native_hw": [source.height, source.width],
                  "profile_hw": list(source.padded), "outputs": args.outputs, "png": args.image_sequence,
                  "compile": args.compile, "hann_chunk": args.hann_chunk, "hann_stride": args.hann_stride,
                  "model_load_seconds": model_load_seconds, "warmups": [], "runs": [],
                  "memory_method": "proc_pid_rusage physical footprint, 20 ms sampling during each workflow",
                  "timing_scope": "decode through completed output writes and scratch cleanup; excludes initial model load"}
        def save_report():
            if args.benchmark_json:
                args.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.benchmark_json.with_suffix('.tmp')
                temporary.write_text(json.dumps(report, indent=2) + '\n')
                temporary.replace(args.benchmark_json)
        save_report()
        for index in range(args.warmup_runs + args.benchmark_runs):
            warmup = index < args.warmup_runs
            timings = {} if args.benchmark_json else None
            if args.benchmark_json:
                from .benchmarking import Footprint
                monitor = Footprint()
            else:
                monitor = nullcontext()
            with monitor as memory:
                started = time.perf_counter()
                # Own only this run's scratch directory, including on failure.
                with tempfile.TemporaryDirectory(prefix=".corridorkey-", dir=args.output_dir) as scratch:
                    run(model, source, args.output_dir, scratch, window=args.hann_chunk, stride=args.hann_stride,
                        outputs=args.outputs, png=args.image_sequence, despill=args.despill_strength,
                        screen=args.screen, first_hint=args.hint_first_frame,
                        hint_source=Source(args.hint_video) if args.hint_video else None, timings=timings)
                seconds = time.perf_counter() - started
            row = {"seconds": seconds, "stage_seconds": timings,
                   "peak_physical_bytes": memory.peak if memory else None}
            report["warmups" if warmup else "runs"].append(row)
            save_report()
        report["status"] = "complete"
        report["median_seconds"] = statistics.median(x["seconds"] for x in report["runs"])
        report["peak_physical_bytes"] = max((x["peak_physical_bytes"] or 0) for x in report["runs"])
        save_report()
    except (ImportError, FileNotFoundError, ValueError) as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
