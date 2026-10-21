# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Directly compare distributed FP32 tensors with the pinned official checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

OFFICIAL_SHA256 = "c43596de98cc8866d577c8a9f9ea239acdf3f7a130ff24e25db4375a50590436"
REVISION = "cf44a8db3f024bb49fd59daa408848ccf05f7477"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True, help="Official corridorkey_v2.safetensors")
    parser.add_argument("--model-root", type=Path, required=True, help="Distributed MLX model root")
    parser.add_argument("--output", type=Path, required=True, help="JSON verification receipt")
    args = parser.parse_args()
    with args.official.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != OFFICIAL_SHA256:
        parser.error("Official checkpoint SHA-256 differs from the verified parent revision")
    rows = []
    with safe_open(args.official, framework="numpy") as official:
        for profile in ("1024x1536-t1", "768x1280-t1"):
            for stage in ("stage1", "stage2"):
                with safe_open(args.model_root/profile/stage/"weights.safetensors", framework="numpy") as exported:
                    keys = set(exported.keys())
                    for name in sorted(k for k in official.keys() if k.startswith(stage+".")):
                        suffix = name.removeprefix(stage+".").replace(".", "_")
                        suffix = ("stage_" if stage == "stage1" else "") + suffix
                        matches = [prefix+suffix for prefix in ("p_", "b_") if prefix+suffix in keys]
                        if len(matches) != 1:
                            raise ValueError(f"Expected one exported match for {profile}: {name}")
                        a, b = official.get_tensor(name), exported.get_tensor(matches[0])
                        equal = (a.shape == b.shape and a.dtype == b.dtype and
                                 np.array_equal(np.ascontiguousarray(a).view(np.uint8).reshape(-1),
                                                np.ascontiguousarray(b).view(np.uint8).reshape(-1)))
                        rows.append({"profile": profile, "official": name, "exported": matches[0],
                                     "direct_byte_equal": bool(equal)})
    passed = len(rows) == 1476 and all(r["direct_byte_equal"] for r in rows)
    report = {"upstream": "nikopueringer/CorridorKey2.0", "revision": REVISION,
              "official_sha256": digest, "comparisons": len(rows), "passed": passed,
              "method": "Exact dtype, shape and equality of uint8 tensor buffers; no numeric tolerance.",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
