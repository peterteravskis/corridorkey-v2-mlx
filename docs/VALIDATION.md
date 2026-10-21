# Release validation

[Home](../README.md) · [Benchmarks](BENCHMARKS.md) · [Precision](PRECISION.md)

Validated on Apple M4 Pro with 48 GiB unified memory, Python 3.12, MLX 0.32.4,
and PyTorch 2.9.1. These checks exercise the independent packaged CLI and
the actual staged model assets. Fresh CLI timing, memory, and output agreement
are recorded in [BENCHMARKS.md](BENCHMARKS.md).

| Check | Observed result |
| :--- | :--- |
| Unit suite | **21 tests passed**, including graph kernels, rewrite precision, recurrent batching, temporal coverage, profile selection, and input validation |
| Fresh CLI benchmark matrix | **30 conditions**, one warm-up plus three measured runs; **864 processed frames**, all completed |
| Kernel output agreement | **432 PNG comparisons passed** across four stills and a 32-frame moving clip at all three precisions |
| FP32 still, 1536 × 1024 | All five 16-bit PNG outputs exactly match the reference MLX workflow: alpha, foreground, raw foreground, checker, cutout |
| FP32 temporal, six repeated frames, Hann 4/2 | **30 / 30 PNG outputs exactly match**, maximum difference zero |
| FP16 temporal with staged models | Six frames, all five outputs: **30 valid 16-bit PNGs**; scratch removed |
| Q8 temporal with staged models | Six frames, all five outputs: **30 valid 16-bit PNGs**; scratch removed |
| Green-screen squirrel | Marker-free input; fresh FP32 CLI generated the displayed checker and unaltered RGBA PNG |
| Blue-screen redhead | FP32 CLI with `--screen blue` generated the displayed checker and unaltered RGBA PNG |
| Video writer | Six 1536 × 1024 frames at 24 fps; H.264 checker and ProRes 4444 cutout decoded by FFprobe; alpha-bearing pixel format confirmed |
| Wheel | Built successfully; isolated imports of CLI, pipeline, and MLX backend succeeded without importing upstream `corridorkey` |
| License metadata | Wheel records `Apache-2.0`, with LICENSE and NOTICE included |
| Model staging | Every distributed graph, statistics file, and weight file matches its source SHA-256 |
| Official parent | Direct byte comparisons: 738 learned tensors and 6 normalization tensors match; both FP32 profiles add 1,476 exact exported-tensor comparisons |

The six-frame fixture exercises recurrence, overlapping windows, and output
ordering; it repeats a still and does not establish real-motion quality.
FP16/Q8 release runs are inference/export smoke tests, not a claim of equality
to FP32. The video check covers the writer with real rendered frames; it does
not establish color-managed HDR or audio support. Other hardware, long clips,
and additional profiles require separate evaluation.

Compact machine-readable receipts are in [benchmarks/release](../benchmarks/release).
The model-byte manifest and parent verification are in the
[model repository](https://huggingface.co/cdelta/CorridorKey-v2-MLX).

Run the unit suite from a source checkout:

```bash
python -m unittest discover -s tests -v
```

Recreate the two README examples after downloading model profiles:

```bash
corridorkey-mlx --compile --image-sequence \
  --model-root /path/to/CorridorKey-v2-MLX \
  --input assets/examples/squirrel-input.png --output-dir outputs/squirrel \
  --outputs alpha checker cutout

corridorkey-mlx --compile --image-sequence --screen blue \
  --model-root /path/to/CorridorKey-v2-MLX \
  --input assets/examples/redhead-input.png --output-dir outputs/redhead \
  --outputs alpha checker cutout
```
