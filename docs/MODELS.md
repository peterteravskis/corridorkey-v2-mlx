# Model profiles

[Home](../README.md) · [Hugging Face model card](https://huggingface.co/cdelta/CorridorKey-v2-MLX)

Choose a precision from the [Hugging Face Collection](https://huggingface.co/collections/cdelta/corridorkey-v2-mlx-6ad84231314d2593c3799e34):

| Precision | Repository | CLI option |
| :--- | :--- | :--- |
| FP32 | [CorridorKey-v2-MLX-fp32](https://huggingface.co/cdelta/CorridorKey-v2-MLX-fp32) | `--precision fp32` (default) |
| Mixed FP16 | [CorridorKey-v2-MLX-fp16](https://huggingface.co/cdelta/CorridorKey-v2-MLX-fp16) | `--precision fp16` |
| Q8 / 8-bit | [CorridorKey-v2-MLX-8bit](https://huggingface.co/cdelta/CorridorKey-v2-MLX-8bit) | `--precision q8` |

```bash
hf download cdelta/CorridorKey-v2-MLX-fp32 --local-dir /path/to/CorridorKey-v2-MLX
corridorkey-mlx --model-root /path/to/CorridorKey-v2-MLX --list-profiles

# Alternatively, download and inspect the 8-bit profiles.
hf download cdelta/CorridorKey-v2-MLX-8bit --local-dir /path/to/CorridorKey-v2-MLX-8bit
corridorkey-mlx --model-root /path/to/CorridorKey-v2-MLX-8bit --precision q8 --list-profiles
```

Each precision repository includes only its own profiles and a matching
`model-manifest.json` and `SHA256SUMS`, plus licenses and shared evidence.
Keep the folder layout intact. Pass the download directory to `--model-root`;
the CLI adds the `fp16/` or `q8/` subfolder when you select that precision.
When downloading several repositories into one directory, common documentation
and manifests are replaced by the last download; use separate directories to
retain each release's complete checksum inventory.

The [combined bundle](https://huggingface.co/cdelta/CorridorKey-v2-MLX) remains available under its existing name:

```bash
hf download cdelta/CorridorKey-v2-MLX --local-dir /path/to/CorridorKey-v2-MLX
```

Combined layout (each separate release contains only its precision's profiles):

```text
CorridorKey-v2-MLX/
├── 1024x1536-t1/
│   ├── stats.json
│   ├── provenance.json
│   ├── rvm/          graph.json + weights.safetensors
│   ├── foundations/  graph.json + weights.safetensors
│   ├── stage1/       graph.json + weights.safetensors
│   └── stage2/       graph.json + weights.safetensors
├── 1024x1536-t4/
│   └── stage1/       graph.json + weights.safetensors
├── 768x1280-t1/     complete FP32 single-frame profile
├── 768x1280-t4/     FP32 stage-one temporal profile
├── fp16/            mixed FP16 profiles, including 768x1280
├── q8/              packed Q8 / FP16 profiles, including 768x1280
├── model-manifest.json
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

`H×W-tN` names the padded height, width, and component frame count. Foundation
and RVM inference use one-frame profiles. Stage two uses batch one. Stage one
uses a one-frame profile for a still or a four-frame profile for temporal clips.
Profiles are static; dimensions must match the graph's declared shapes.

FP32 is the default. Select `--precision fp16` or `--precision q8` to use those
subfolders. Both include complete 1024×1536 and 768×1280 single-frame profiles
and four-frame stage-one profiles. Their `768x1280-t2` profile is stage two only;
it does not enable two-frame temporal windows. Additional stage-two batch profiles
are included for API/research use; the CLI uses batch one.

For 1280×720 footage with any of the three precisions, use `--height 768 --width 1280`.
For a Q8-only download with complete matching checksums, use the dedicated
`cdelta/CorridorKey-v2-MLX-8bit` repository.
The converted graph format is `corridorkey-mlx-graph-v1`, not a Transformers
`AutoModel` or MLX-LM checkpoint.

The original parent is
[`nikopueringer/CorridorKey2.0`](https://huggingface.co/nikopueringer/CorridorKey2.0),
pinned for verification at `cf44a8db3f024bb49fd59daa408848ccf05f7477`.
The conversion input's 738 learned tensors and all six foundation-normalization
tensors match the official release byte for byte, with identical shapes and
dtypes. See the companion repository's `weight-verification.json` and
`weight-tensor-hashes.json` for the original verification results. The renewed
`current-hf-bytecheck.json` records direct byte comparisons against the current
public revision. `fp32-export-bytecheck.json` confirms all 1,476 stage-one and
stage-two tensor comparisons in the two distributed FP32 single-frame profiles.

Model graphs and weights retain the original model terms. The Apache 2.0
source license does not cover them. Conversion/training tools from upstream
must be obtained and used under their own license; they are not vendored here.

## Recheck the distributed FP32 tensors

Download the pinned official checkpoint to your model storage, then run:

```bash
hf download nikopueringer/CorridorKey2.0 corridorkey_v2.safetensors \
  --revision cf44a8db3f024bb49fd59daa408848ccf05f7477 \
  --local-dir /path/to/official-model
python scripts/verify_parent.py \
  --official /path/to/official-model/corridorkey_v2.safetensors \
  --model-root /path/to/CorridorKey-v2-MLX --output fp32-bytecheck.json
```

The verifier first checks the official file SHA-256, then directly compares
tensor bytes, dtypes, and shapes in both distributed FP32 single-frame profiles.
It does not use a floating-point tolerance. FP16 and Q8 are intentionally excluded.
