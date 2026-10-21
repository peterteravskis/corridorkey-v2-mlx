# Command-line guide

[Home](../README.md) · [Models](MODELS.md) · [Python API](API.md)

Run `corridorkey-mlx --help` or `python -m corridorkey_mlx --help`.
The compatibility entry point `python infer_mlx.py` accepts the same arguments.
Legacy underscore spellings remain available for the original CLI options.

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--input` | Required | PNG/JPEG still, directory of PNG/JPEG frames, or video |
| `--output-dir` | Required for inference | Separate destination directory |
| `--model-root` | `models` | Root containing `1024x1536-t1`, etc. |
| `--precision` | `fp32` | `fp32` at the root, or `fp16` / `q8` subfolder |
| `--height`, `--width` | `1024`, `1536` | Padded profile dimensions, multiples of 128 |
| `--compile` | Off | Compile the MLX execution graphs |
| `--kernel-mode` | `enabled` | Release optimizations, or `reference` to disable custom rewrites/kernels for a matched comparison |
| `--benchmark-json` | None | Save measured workflow times, stage timings and sampled physical footprint |
| `--benchmark-runs` | `1` | Number of measured workflows using the same loaded model |
| `--warmup-runs` | `0` | Complete warm-up workflows before measurement |
| `--hann-chunk`, `--hann-stride` | `4`, `2` | Stage-one temporal window and step |
| `--image-sequence` | Off | Write 16-bit PNGs instead of preview videos/MOV |
| `--outputs` | `alpha fg checker cutout` | Space-separated outputs; add `raw_fg` if wanted |
| `--screen` | `green` | Green or blue despill; no input-channel remapping |
| `--despill-strength` | `1` | Cleanup strength in `[0, 1]`; alpha is unchanged |
| `--hint-first-frame` | None | Matte image for the first frame; alpha channel preferred |
| `--hint-video` | None | Full hint sequence/video matching source frame count and dimensions |
| `--threads` | `8` | CPU preparation threads |
| `--check` | Off | Inspect input and required profile files without loading weights |
| `--list-profiles` | Off | List available component/profile directories |
| `--low-vram` | Off | Compatibility flag; the independent scheduler always uses disk-backed staging |

For a short clip without a two-/three-frame stage-one profile:

```bash
corridorkey-mlx --compile --image-sequence \
  --input short.mp4 --output-dir outputs/short --model-root /path/to/models \
  --hann-chunk 1 --hann-stride 1
```

This removes temporal-window context; it is an explicit tradeoff, not equivalent
to a four-frame window. RVM recurrence is still carried through the clip.

Image sequences are sorted lexicographically; hidden files such as macOS
`._` sidecars are ignored. Use zero-padded names such as
`000001.png`. Frame-directory video export currently uses 24 fps. Video input
uses its reported frame rate. Outputs contain no audio. All frames must share
one size. PNG/JPEG inputs may be 8-bit or 16-bit; EXR and color-metadata
conversion are outside this basic CLI's supported surface.

The pipeline preserves native dimensions by symmetric reflection padding and
cropping. It normalizes display-referred RGB, applies the model's input encoding,
retains recurrent state, blends overlapping temporal outputs, and refines one
frame at a time. Full-frame hints are used where there is no carried auxiliary
hint; a first-frame hint takes precedence on frame zero. Hint confidence is 0.95.

Each run owns a temporary scratch directory inside its output directory. Feature
and input staging can consume substantial disk space for long clips. Scratch is
removed on normal completion and Python exceptions; forced termination or power
loss can leave a `.corridorkey-*` directory. Use a destination with sufficient
space. Existing named output files may be replaced; choose a new directory for
each run. Input/output directories may not contain one another.

## Reproduce a CLI benchmark

```bash
corridorkey-mlx --compile --image-sequence \
  --input assets/examples/squirrel-input.png --output-dir outputs/benchmark \
  --model-root /path/to/CorridorKey-v2-MLX --precision fp32 \
  --kernel-mode enabled --warmup-runs 1 --benchmark-runs 3 \
  --benchmark-json outputs/benchmark.json
```

Run again with `--kernel-mode reference` in a separate process and destination
to compare kernels at the same precision. For a native 1280×720 clip, use
`--height 768 --width 1280`. Keep the inputs, temporal options and output types
identical between conditions.

Each measured workflow includes decoding, feature staging, inference, rendering,
completed file writes, and scratch cleanup. Initial model construction is
reported separately; the first warm-up includes graph compilation. Repeated
runs overwrite the same named outputs and reuse the loaded model. The JSON
records all samples, per-stage call durations, and process physical footprint
sampled every 20 ms. Interpreter/import startup is excluded. Filesystem caches
are not flushed between runs.

For an application that needs custom scheduling, load [the core API](API.md)
and use your own disk/memory policy.
