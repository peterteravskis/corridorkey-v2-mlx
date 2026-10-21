# CLI benchmarks

[Home](../README.md) · [Raw receipts](../benchmarks/cli/summary.json) · [Release validation](VALIDATION.md)

Measured 2026-10-20 on Apple M4 Pro, 48 GiB, Python 3.12, MLX 0.32.4,
PyTorch 2.9.1. This study runs the packaged `python -m corridorkey_mlx` CLI
in a fresh process for each fixture, precision, and kernel mode.

## Design and timing endpoint

- Four 1536 × 1024 stills: green squirrel, blue redhead/gauze, green hair/gauze,
  and blue fur/glass; one 32-frame moving clip at 1280 × 720, padded to 1280 × 768.
- FP32, FP16, Q8; reference and enabled kernels: **30 conditions**.
- One full warm-up and three measured runs per condition. Precision order
  rotates and mode order alternates; inference processes run sequentially.
- All runs decode, infer, render, and write alpha, foreground, checker, and
  cutout as lossless 16-bit PNGs. Scratch cleanup is inside the timed endpoint.
- Model construction is measured separately. First workflow includes lazy
  profile loading and compilation; warmed runs exclude that first-use work.
  Python interpreter/import startup is excluded from plotted timings.
- Hann window/stride 4/2, stage-two batch one, compilation enabled. Stills use
  the single-frame path. Physical footprint is sampled every 20 ms with macOS
  `proc_pid_rusage`; this is process physical memory, not an allocator counter.

## Moving-clip results

| Precision | Reference | Enabled | Less time | Enabled physical memory |
| :--- | ---: | ---: | ---: | ---: |
| FP32 | 50.465 s | 37.657 s | **25.4%** | 7.05 GiB |
| FP16 | 42.975 s | 37.123 s | **13.6%** | 4.11 GiB |
| Q8 | 46.088 s | 39.397 s | **14.5%** | 3.28 GiB |

![Reference and enabled kernel timings](../assets/cli-kernel-gains.png)
![Workflow and physical memory across fixtures](../assets/cli-workflow.png)
![Stage breakdown](../assets/cli-stage-times.png)
![First-use timing](../assets/cli-first-use.png)

## Output checks

**432 saved PNGs** were compared across 108 frames: 36 frames × three precisions
× four outputs. Each optimized path is compared with its own precision's
reference. Alpha max error ≤ 0.002 and MAE ≤ 0.00002; foreground/checker max
≤ 0.005 and MAE ≤ 0.00002. Foreground RGB uses pixels where reference alpha
exceeds 0.01. RGBA alpha is checked separately with the alpha limits.

All comparisons passed. Receipts preserve pixel-exact flags, maximum error,
MAE, and hashes for every compared file. FP32 rounding differences are not
described as exact equality. This study measures implementation agreement;
there are no ground-truth mattes and no perceptual-quality score.

![Maximum saved alpha and checker code differences](../assets/cli-output-agreement.png)

The marker-free squirrel is the fixture used for this rerun. Input and source
hashes are recorded in [summary.json](../benchmarks/cli/summary.json). Per-condition
JSON files retain all timing samples and the measured command configuration.
The [CLI guide](CLI.md) gives the benchmark command; rerun with each precision
and both `--kernel-mode reference` and `--kernel-mode enabled`.

Regenerate the figures from the receipts:

```bash
python scripts/render_cli_benchmarks.py
```

Figure rendering needs Matplotlib and NumPy. Both PNG and SVG are generated.
The retained [historical MPS study](HISTORICAL_BENCHMARKS.md) and
[isolated operator study](HISTORICAL_OPTIMIZATIONS.md) have different endpoints
and baselines. No cumulative speedup is inferred from their ratios.
