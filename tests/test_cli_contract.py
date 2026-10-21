# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Preflight catches incompatible profiles before any neural model is loaded."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from corridorkey_mlx.cli import main, required_files
from corridorkey_mlx.pipeline import Source, temporal_plan


class CLIContractTests(unittest.TestCase):
    def test_benchmark_runs_and_reference_mode_are_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plate = root / 'plate.png'
            cv2.imwrite(str(plate), np.zeros((128, 128, 3), np.uint8))
            for path in required_files(root, 128, 128, 1):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"precision":"fp32"}')
            report = root / 'benchmark.json'
            args = ['--input', str(plate), '--output-dir', str(root/'out'), '--model-root', str(root),
                    '--height', '128', '--width', '128', '--image-sequence', '--kernel-mode', 'reference',
                    '--warmup-runs', '1', '--benchmark-runs', '3', '--benchmark-json', str(report)]
            with patch('corridorkey_mlx.mlx_backend.MLXModel') as model, patch('corridorkey_mlx.pipeline.run') as run:
                self.assertEqual(main(args), 0)
                self.assertEqual(model.call_args.kwargs['optimizations'], ())
                self.assertEqual(run.call_count, 4)
            data = json.loads(report.read_text())
            self.assertEqual(data['status'], 'complete')
            self.assertEqual(len(data['warmups']), 1)
            self.assertEqual(len(data['runs']), 3)
            self.assertFalse(list((root/'out').glob('.corridorkey-*')))

    def test_precision_selects_and_validates_matching_profiles(self):
        for precision, declaration in (("fp32", "fp32"), ("fp16", "fp16-mixed-v1"), ("q8", "q8-fp16-v1")):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                plate = root / "plate.png"
                cv2.imwrite(str(plate), np.zeros((128, 128, 3), np.uint8))
                selected = root if precision == "fp32" else root / precision
                for path in required_files(selected, 128, 128, 1):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"precision": declaration}))
                args = ["--check", "--image-sequence", "--input", str(plate),
                        "--height", "128", "--width", "128", "--model-root", str(root),
                        "--precision", precision]
                with patch("corridorkey_mlx.mlx_backend.MLXModel") as model:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(args), 0)
                    model.assert_not_called()
                (selected / "128x128-t1/stage1/graph.json").write_text('{"precision":"wrong"}')
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)

    def test_temporal_windows_cover_every_frame_with_positive_weight(self):
        for count in range(1, 18):
            for window, stride in ((1, 1), (4, 2), (4, 4), (4, 1)):
                sums = np.zeros(count)
                plan = temporal_plan(count, window, stride)
                self.assertEqual(plan[-1][0] + len(plan[-1][1]), count)
                for start, weights in plan:
                    sums[start:start + len(weights)] += weights.numpy()
                self.assertTrue(np.all(sums > 0))

    def test_changing_sequence_dimensions_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root / "000.png"), np.zeros((128, 128, 3), np.uint8))
            cv2.imwrite(str(root / "001.png"), np.zeros((128, 129, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "changing frame dimensions"):
                list(Source(root).frames())

    def test_macos_sidecar_files_are_not_treated_as_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cv2.imwrite(str(root/'000.png'), np.zeros((128,128,3), np.uint8))
            (root/'._000.png').write_bytes(b'not an image')
            source = Source(root)
            self.assertEqual(source.count, 1)
            self.assertEqual(len(list(source.frames())), 1)


if __name__ == "__main__":
    unittest.main()
