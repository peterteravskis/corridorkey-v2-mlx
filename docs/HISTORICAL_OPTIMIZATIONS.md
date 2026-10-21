# Speed optimizations included in this release

[Home](../README.md) · [Precision choices](PRECISION.md) · [Raw results](../benchmarks/)

The model loader enables these automatically through `MLXGraph`:

| Path | Enabled work |
| :--- | :--- |
| FP32 | Reconstruction kernel; reflection padding and layout handling; dense projection specialization/pairing; paired 7×7 channel convolution |
| FP16 / Q8 | Paired 7×7 channel convolution that preserves half-precision rounding |

The default option set is `reconstruction`, `padding`, `layout`, `dense`,
`paired_dense`, `channel_conv7`, `half_channel_conv7`. Precision guards select
the applicable kernels. The `half_paired_dense` experiment is not enabled.
Experimental half reconstruction/padding substitutions remain disabled.

These optimizations retain the selected precision, resolution, and temporal
context. They do not substitute a smaller network. Batch one remains the CLI
default because the renewed batch-two screen did not improve throughput.

## Measured 32-frame workflow improvement

| Precision | Before | After | Less time | Output agreement |
| :--- | ---: | ---: | ---: | :--- |
| FP32 | 39.502 s | 33.151 s | **16.1%** | Exact against preceding FP32 path |
| FP16 | 40.981 s | 33.918 s | **17.2%** | Exact against preceding FP16 path |
| Q8 | 42.074 s | 34.313 s | **18.4%** | Exact against preceding Q8 path |

![Measured optimization gains](../assets/optimization-gains.png)

M4 Pro, 48 GiB, native 1280×720 video padded to 1280×768. Each comparison
uses one warm-up and three measured runs, with decode through uint16 rendering;
PNG writing is excluded. The FP32 and half-precision studies are separate
before/after experiments. Their absolute times are not a controlled three-way
precision ranking. All 36 frames per precision (32 moving frames and four
stills) agree exactly with the corresponding previous implementation; the
video adds zero temporal alpha error.

FP32 still-time reductions span 4.3–13.4%. Dense packing contributes modestly;
the specialized convolution supplies most of the gain. Video physical footprint
is essentially unchanged for FP32 (7.23 → 7.30 GiB). The optimizations do not
erase the precision differences between FP32, FP16, and Q8.

Evidence: [FP32 results](../benchmarks/dense-conv-results.json),
[FP16/Q8 results and batch screen](../benchmarks/half-dense-conv-results.json).
These records describe the development benchmark harness. The independent
CLI's correctness validation is [reported separately](VALIDATION.md).

## Isolated kernel measurements

![Measured latency and individual samples for reconstruction, reflection padding, and paired convolution](../assets/kernel-latency.png)

These measurements compare compiled operators on fixed inputs. Bars show the
median, with every recorded timing sample shown as a dot. Reconstruction and
padding have five alternating samples per implementation; convolution has
seven. Each panel uses its own millisecond scale.

| Kernel / shape | Reference median | Optimized median | Ratio |
| :--- | ---: | ---: | ---: |
| Reconstruction · batch 1 | 110.666 ms | 8.213 ms | 13.47× |
| Reconstruction · batch 4 | 404.733 ms | 42.011 ms | 9.63× |
| Reflection padding · 1280 channels | 1.454 ms | 0.307 ms | 4.73× |
| Reflection padding · 3840 channels | 8.435 ms | 0.812 ms | 10.38× |
| Paired 7×7 convolution · batch 1, 1280 channels | 10.548 ms | 2.859 ms | 3.69× |
| Paired 7×7 convolution · batch 4, 1280 channels | 40.348 ms | 10.268 ms | 3.93× |

The plot also includes the small controls, where the gains are much smaller.
Maximum absolute output difference is **9.54e−6 for reconstruction**, and
**zero for reflection padding and paired convolution**. These are isolated
operator ratios; the measured whole-workflow improvements are reported above.

Data: [reconstruction and padding](../benchmarks/isolated-kernels.json),
[paired convolution](../benchmarks/channel-conv.json).

## Controlled FP16/Q8 kernel selection

![Stage-two ablation separates projection pairing, convolution, and both changes](../assets/kernel-ablation.png)

The compiled 768×1280 stage-two graph was measured with batch one, fixed-seed
synthetic inputs, two warm-ups, and five measured calls per candidate. Candidate
order rotated between runs. Convolution alone reduced FP16 stage time from
**642.2 ms to 551.7 ms**, and Q8 from **663.6 ms to 576.8 ms**. Each candidate's
raw outputs matched its precision baseline exactly.

Projection pairing alone was slower. Combining it with convolution was also
slower than convolution alone, which is why `half_paired_dense` stays disabled.
This graph test isolates the selected kernel's contribution; the moving-video
comparison above checks that it also improves the complete measured workflow.

Data: [all candidates and timing samples](../benchmarks/half-operator-screen.json).
