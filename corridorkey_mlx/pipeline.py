# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Independent file I/O and temporal scheduling for exported MLX graphs.

This module implements the published tensor interface; it imports no upstream
CorridorKey Python code. Inputs are display-referred RGB. Scratch files bound
RAM use independently of clip length and are owned by the calling context.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class Source:
    path: Path
    fps: float = 24.

    def __post_init__(self):
        self.path = Path(self.path)
        self.images = (sorted(p for p in self.path.iterdir() if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in IMAGE_EXTENSIONS)
                       if self.path.is_dir() else
                       [self.path] if self.path.suffix.lower() in IMAGE_EXTENSIONS else None)
        if self.images is not None:
            if not self.images:
                raise ValueError(f"No PNG/JPEG frames in {self.path}")
            first = cv2.imread(str(self.images[0]), cv2.IMREAD_UNCHANGED)
            self.count = len(self.images)
        else:
            capture = cv2.VideoCapture(str(self.path))
            try:
                ok, first = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode {self.path}")
                self.count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            finally:
                capture.release()
        if first is None or self.count < 1:
            raise ValueError(f"Cannot read frames from {self.path}")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Frame rate must be finite and positive")
        self.height, self.width = first.shape[:2]
        self.padded = tuple(((n + 127) // 128) * 128 for n in (self.height, self.width))

    def frames(self):
        capture = None if self.images is not None else cv2.VideoCapture(str(self.path))
        try:
            for index in range(self.count):
                if self.images is not None:
                    raw = cv2.imread(str(self.images[index]), cv2.IMREAD_UNCHANGED)
                else:
                    ok, raw = capture.read()
                    if not ok:
                        raise ValueError(f"Video ended before frame {index}/{self.count}")
                if raw is None or raw.shape[:2] != (self.height, self.width):
                    raise ValueError(f"Unreadable frame or changing frame dimensions at {index}")
                yield raw
        finally:
            if capture is not None:
                capture.release()


def unit_pixels(raw):
    if raw.dtype == np.uint8:
        return raw.astype(np.float32) / 255.
    if raw.dtype == np.uint16:
        return raw.astype(np.float32) / 65535.
    raise ValueError("This SDR CLI accepts 8-bit or 16-bit integer images; HDR EXR is not supported")


def pad_image(image, shape):
    h, w = image.shape[:2]
    dh, dw = shape[0] - h, shape[1] - w
    if dh < 0 or dw < 0:
        raise ValueError("Profile cannot be smaller than input")
    return cv2.copyMakeBorder(image, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                              cv2.BORDER_REFLECT_101)


def encode_frame(raw, shape):
    image = unit_pixels(raw)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    rgb = np.ascontiguousarray(image[..., :3][..., ::-1])
    rgb = pad_image(rgb, shape)
    # IEC 61966-2-1 followed by the model's documented signed-asinh encoding.
    tensor = torch.from_numpy(rgb)
    linear = torch.where(tensor <= .04045, tensor / 12.92, ((tensor + .055) / 1.055).pow(2.4))
    encoded = np.arcsinh(linear.numpy() / .008) / np.arcsinh(16. / .008)
    model_input = torch.from_numpy(np.ascontiguousarray(encoded.transpose(2, 0, 1))).bfloat16().float()
    display = tensor.permute(2, 0, 1).bfloat16().float()
    return model_input, display


def temporal_plan(count, window, stride):
    """Return full windows ending at the final frame, with cosine overlap weights."""
    if count < 1 or window < 1 or not 1 <= stride <= window:
        raise ValueError("Invalid temporal configuration")
    length = min(window, count)
    last = count - length
    starts = sorted(set(range(0, last + 1, stride)) | {last})
    overlap = window - stride
    plan = []
    for index, start in enumerate(starts):
        weights = torch.ones(length)
        if overlap and length > 1:
            n = min(length, overlap)
            phase = (torch.arange(n, dtype=torch.float32) + 1) * math.pi / (overlap + 1)
            ramp = (1 - torch.cos(phase)) * .5
            if index:
                weights[:n] = ramp
            if index + 1 < len(starts):
                weights[-n:] = ramp.flip(0)
        plan.append((start, weights))
    return plan


def checkerboard(height, width):
    y, x = np.indices((height, width))
    tile = np.where(((x // 64 + y // 64) % 2) == 0, .75, .5).astype(np.float32)
    return torch.from_numpy(np.broadcast_to(tile, (3, height, width)).copy())


class Outputs:
    def __init__(self, destination, names, source, png):
        self.root, self.names, self.source, self.png = Path(destination), sorted(names), source, png
        self.processes = {}
        self.root.mkdir(parents=True, exist_ok=True)
        if not png and shutil.which("ffmpeg") is None:
            raise FileNotFoundError("Install ffmpeg for video export, or use --image-sequence")

    def write(self, index, rendered):
        for name, batch in rendered.items():
            pixels = np.ascontiguousarray(batch[0])
            if self.png:
                folder = self.root / name
                folder.mkdir(exist_ok=True)
                target = folder / f"{index:06d}.png"
                if not cv2.imwrite(str(target), pixels):
                    raise OSError(f"Failed to write {target}")
            else:
                if name not in self.processes:
                    alpha = name == "cutout"
                    target = self.root / (name + (".mov" if alpha else ".mp4"))
                    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                            "-pixel_format", "bgra64le" if alpha else "bgr48le", "-video_size",
                            f"{self.source.width}x{self.source.height}", "-framerate", str(self.source.fps),
                            "-i", "pipe:0", "-an"]
                    args += (["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le"]
                             if alpha else ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv444p"])
                    self.processes[name] = subprocess.Popen(args + [str(target)], stdin=subprocess.PIPE)
                self.processes[name].stdin.write(pixels.tobytes())

    def close(self):
        failed = []
        for name, process in self.processes.items():
            try:
                process.stdin.close()
            except BrokenPipeError:
                failed.append(name)
            if process.wait():
                failed.append(name)
        if failed:
            raise RuntimeError(f"ffmpeg export failed: {', '.join(sorted(set(failed)))}")


@torch.no_grad()
def run(model, source, destination, scratch, *, window=4, stride=2,
        outputs=("alpha", "fg", "checker", "cutout"), png=False,
        despill=1., screen="green", first_hint=None, hint_source=None, timings=None):
    """Disk-backed features; sequential RVM; overlapping stage one; stage two batch one."""
    scratch = Path(scratch)
    def timed(name, function, *args, **kwargs):
        if timings is None:
            return function(*args, **kwargs)
        started = time.perf_counter()
        result = function(*args, **kwargs)
        timings[name] = timings.get(name, 0.) + time.perf_counter() - started
        return result
    h, w = source.padded
    if hint_source and (hint_source.count != source.count or
                        (hint_source.height, hint_source.width) != (source.height, source.width)):
        raise ValueError("Hint sequence must have the same frame count and dimensions as the input")

    def hint_tensor(raw):
        values = unit_pixels(raw)
        if values.ndim == 3:
            values = values[..., 3] if values.shape[2] == 4 else values[..., 0]
        if values.shape != (source.height, source.width):
            raise ValueError("Hint dimensions must match input")
        return torch.from_numpy(pad_image(values, (h, w)).copy())[None].bfloat16().float()

    initial = None
    if first_hint:
        raw = cv2.imread(str(first_hint), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Cannot decode hint {first_hint}")
        initial = hint_tensor(raw)
    hints = hint_source.frames() if hint_source else None
    state = None
    mean = model.foundations.rvm_mean.reshape(1, 1, -1, 1, 1)
    std = model.foundations.rvm_std.reshape(1, 1, -1, 1, 1).clamp_min(1e-6)
    try:
        for index, raw in enumerate(source.frames()):
            rgb, display = encode_frame(raw, (h, w))
            small = F.interpolate(display[None], size=(h // 4, w // 4), mode="bilinear", align_corners=False)
            recurrent, state = timed('rvm', model.foundations.rvm, small[None].clamp(0, 1), state)
            features = timed('foundations', model.foundations,
                F.interpolate(display[None], size=(h // 4, w // 4), mode="area"),
                F.interpolate(display[None], size=(h * 7 // 32, w * 7 // 32), mode="area"))
            payload = dict(rgb=rgb.numpy(), radio=features["cradio"][0].numpy(),
                           moge=features["moge"][0].numpy(),
                           rvm=((recurrent - mean) / std)[0, 0].numpy().astype(np.float16))
            if hints:
                payload["hint"] = hint_tensor(next(hints)).numpy()
            np.savez(scratch / f"frame-{index:06d}.npz", **payload)
            print(f"Features {index + 1}/{source.count}", flush=True)
    finally:
        if hints:
            hints.close()

    meta = {"model_native_hw": (source.height, source.width), "output_hw": (source.height, source.width),
            "pad": ((w - source.width) // 2, 0, (h - source.height) // 2, 0),
            "despill_strength": despill, "screen": screen}
    checker = checkerboard(source.height, source.width)
    plan = temporal_plan(source.count, window, stride)
    accumulated, denominators, carry = {}, {}, {}
    writer = Outputs(destination, outputs, source, png)
    try:
        for window_index, (start, weights) in enumerate(plan):
            indices = list(range(start, start + len(weights)))
            records = []
            for index in indices:
                with np.load(scratch / f"frame-{index:06d}.npz", allow_pickle=False) as data:
                    records.append({key: data[key] for key in data.files})
            def stack(key):
                return torch.from_numpy(np.stack([record[key] for record in records])).float()[None]
            hint = torch.zeros(1, len(indices), 1, h, w)
            confidence = torch.zeros(1, len(indices))
            supplied = False
            for offset, index in enumerate(indices):
                value = (initial if index == 0 and initial is not None else
                         carry.get(index, records[offset].get("hint")))
                if value is not None:
                    hint[0, offset] = torch.as_tensor(value)
                    confidence[0, offset] = .95
                    supplied = True
            if not supplied:
                confidence.fill_(-1)
            result = timed('stage1', model.stage1, stack("rgb"), hint, stack("moge"), stack("radio"), stack("rvm"),
                                  confidence.bfloat16().float())
            aux = F.interpolate(result["aux"][0, :, :1].float(), size=(h, w), mode="bilinear",
                                align_corners=False).clamp(0, 1).bfloat16().float()
            weighted = (result["baton"][0].float() * weights[:, None, None, None]).numpy()
            for offset, index in enumerate(indices):
                carry[index] = aux[offset]
                if index in accumulated:
                    accumulated[index] += weighted[offset]
                    denominators[index] += float(weights[offset])
                else:
                    accumulated[index] = weighted[offset].copy()
                    denominators[index] = float(weights[offset])
            boundary = plan[window_index + 1][0] if window_index + 1 < len(plan) else source.count
            for index in sorted(k for k in accumulated if k < boundary):
                baton = (accumulated.pop(index) / denominators.pop(index)).astype(np.float16)
                with np.load(scratch / f"frame-{index:06d}.npz", allow_pickle=False) as record:
                    rgb = torch.from_numpy(record["rgb"])[None, None]
                prediction = timed('stage2', model.stage2, rgb, torch.from_numpy(baton).float()[None, None])
                rendered = timed('render', model._render_predictions, meta, set(outputs), checker, prediction[0])
                timed('write', writer.write, index, rendered)
                carry.pop(index, None)
                (scratch / f"frame-{index:06d}.npz").unlink()
                print(f"Export {index + 1}/{source.count}", flush=True)
    finally:
        writer.close()
    (Path(destination) / "settings.json").write_text(json.dumps({
        "engine": "corridorkey-v2-mlx", "frames": source.count, "fps": source.fps,
        "native_hw": [source.height, source.width], "profile_hw": [h, w],
        "window": window, "stride": stride, "stage2_batch": 1,
        "outputs": sorted(outputs), "despill_strength": despill, "screen": screen,
        "image_sequence": png, "audio": "not copied", "input_color": "display-referred sRGB",
    }, indent=2) + "\n")
