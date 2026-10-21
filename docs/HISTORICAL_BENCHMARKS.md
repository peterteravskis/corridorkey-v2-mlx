# Measurements and limits

[Home](../README.md) · [Raw records](../benchmarks/) · [Release validation](VALIDATION.md)

These are historical engine-development results on an Apple M4 Pro with 48 GiB,
Python 3.12.8, MLX 0.32.4, and PyTorch 2.9.1. The new independent CLI has a
different I/O/scheduling implementation. Its correctness checks are separate;
we do not claim the historical end-to-end speed for the new CLI.

## MPS versus MLX

Both paths use native 1536 × 1024 inputs, FP32 neural compute, Hann 4/2,
stage-two batch one, and despill strength one. Timings include decoding and
four completed lossless 16-bit PNG exports. Model loading and warm-up are
excluded. Five measured still runs per plate/backend were split into alternating
3+2 blocks, with two warm-ups per process. The six-frame clip used three measured
runs after one warm-up.

| Case | MPS | MLX | Ratio |
| :--- | ---: | ---: | ---: |
| Hair / gauze | 3.59565 s | 2.38573 s | 1.507× |
| Fur / blurred paw | 3.50356 s | 2.29976 s | 1.523× |
| Six-frame fixture | 21.92370 s | 15.52647 s | 1.412× |

The temporal fixture repeats a still. It tests recurrent state, overlapping
windows, and carried hints; it does not establish real-motion matte quality.
The plates are generated evaluation examples, not a labeled ground-truth dataset.

The memory probe samples macOS `proc_pid_rusage` physical footprint every 20 ms
in separate processes. Warmed peaks were 20,673,564,536 bytes for MPS and
7,719,966,456 bytes for MLX. MLX's allocator cache was capped at 256 MiB, while
MPS retained its reference settings. This is a workflow comparison, not a
universal ratio between libraries; RSS and allocator counters are not substituted
for physical footprint.

Fixed agreement limits: alpha MAE ≤ 0.00002 and max ≤ 0.002; visible foreground
and checker MAE ≤ 0.00002 and max ≤ 0.005. Foreground is evaluated where reference
alpha exceeds 0.01. Both stills have alpha/checker differences ≤ 2/65535 and
visible foreground differences ≤ 6/65535. The repeated-still clip's worst alpha
difference is 13/65535. These quantify backend agreement, not ground-truth quality.

Source: [baseline summary](../benchmarks/baseline-summary.json), individual
`*-mlx-*.json` / `*-mps-*.json` records, and
[component parity](../benchmarks/mlx-component-parity.json).

## Later FP32 convolution optimization

A separate comparison against the preceding optimized MLX path used one warm-up
and three measured runs per case/mode. It includes decode through uint16
rendering and excludes PNG writing.

| Case | Previous MLX | Optimized MLX | Less time |
| :--- | ---: | ---: | ---: |
| Moving video · 32 frames | 39.502 s | 33.151 s | 16.1% |
| Hair / gauze | 1.721 s | 1.524 s | 11.5% |
| Blue hair / gauze | 1.687 s | 1.482 s | 12.1% |
| Fur / blurred paw | 1.698 s | 1.471 s | 13.4% |
| Blue fur / glass | 1.690 s | 1.617 s | 4.3% |

All 36 frames match exactly in raw predictions, alpha, foreground, and composites.
The moving-video comparison adds zero temporal alpha error. Peak physical
footprint was 7.23 → 7.30 GiB. See
[full measurements](../benchmarks/dense-conv-results.json).

## Precision experiments

FP16/Q8 paired-convolution changes reduced their respective 32-frame workflow
times by 17.2% / 18.4% relative to their own preceding implementations. They
matched those implementations exactly across 36 frames per precision, but both
precisions fail the stricter FP32 agreement limits. Q8 is a memory tradeoff;
it is not the fastest measured variant. Neither is the default model here.
See [half-precision experiment records](../benchmarks/half-dense-conv-results.json).

Absolute machine paths in the public JSON have been generalized. Numerical
measurements and recorded hashes are preserved; [PROVENANCE.json](../benchmarks/PROVENANCE.json)
records both the original and public-file SHA-256 hashes. Do not multiply
speedups from different endpoints or attribute these results to other Macs.
