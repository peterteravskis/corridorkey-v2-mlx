# Copyright 2026 Peter Teravskis
# SPDX-License-Identifier: Apache-2.0
"""Rebuild the README plots from bundled measurements, without rerunning inference."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, INK, MUTED = "#25282d", "#2d3137", "#edf1f6", "#b6bec9"
COLORS = ["#8299c1", "#67dbba", "#ecb875"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": PANEL,
    "savefig.facecolor": BG, "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": INK, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titleweight": "bold", "axes.titlepad": 16,
    "axes.edgecolor": "#49505a", "grid.color": "#49505a", "grid.alpha": .55,
    "legend.facecolor": BG, "legend.edgecolor": BG, "legend.labelcolor": INK,
    "svg.fonttype": "none"})


def read(name):
    return json.loads((ROOT / "benchmarks" / name).read_text())


def style(ax, label):
    ax.set_xlabel(label, labelpad=10)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)


def header(fig, title, subtitle, footer):
    fig.text(.055, .95, "CORRIDORKEY v2 / MLX", fontsize=11, color=COLORS[1], weight="bold")
    fig.text(.055, .895, title, fontsize=23, weight="bold")
    fig.text(.055, .852, subtitle, fontsize=11, color=MUTED)
    fig.text(.055, .035, footer, fontsize=9, color=MUTED, linespacing=1.55)


def save(fig, stem):
    (ROOT / "assets").mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(ROOT / "assets" / f"{stem}.{ext}", dpi=160)
    plt.close(fig)


def baseline():
    d = read("baseline-summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.subplots_adjust(left=.15, right=.96, bottom=.27, top=.72, wspace=.65)
    header(fig, "Matched backend comparison", "Apple M4 Pro · 48 GiB · FP32 · warmed workflow · lower is better",
           "Historical engine benchmark; includes four completed 16-bit PNG exports. New CLI timings are separate.\n"
           "Stills: five samples/backend. Six-frame repeated-still clip: three samples/backend. Memory: separate physical-footprint probe.")
    for i, (case, label) in enumerate([("hair-gauze", "Hair / gauze"), ("fur-motion", "Fur / blur")]):
        for j, backend in enumerate(["mps", "mlx"]):
            v = d["stills"][case][backend]["median_seconds"]
            y = i * 1.45 + j * .45
            axes[0].barh(y, v, height=.34, color=COLORS[j], label=backend.upper() if i == 0 else None)
            axes[0].text(v + .08, y, f"{v:.2f}", va="center", fontsize=10)
    axes[0].set_yticks([.225, 1.675], ["Hair / gauze", "Fur / blur"])
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 4.5)
    axes[0].set_title("Still image"); style(axes[0], "seconds / frame")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center",
               bbox_to_anchor=(.24, .125), ncol=2, fontsize=10)
    clip = d["video"]["video-six"]
    values = [clip[x]["median_seconds"] for x in ("mps", "mlx")]
    memory = [d["memory"][x]["peak_after_warmup_bytes"] / 2**30 for x in ("mps", "mlx")]
    for ax, vals, title, xlabel in [(axes[1], values, "Six-frame clip", "seconds / clip"),
                                     (axes[2], memory, "Physical footprint", "GiB · warmed peak")]:
        ax.barh([0, 1], vals, color=COLORS[:2], height=.45)
        ax.set_yticks([0, 1], ["MPS", "MLX"]); ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.3)
        for y, v in enumerate(vals): ax.text(v + max(vals)*.035, y, f"{v:.2f}", va="center")
        ax.set_title(title); style(ax, xlabel)
    save(fig, "mlx-mps-performance")


def optimization():
    a = next(x for x in read("dense-conv-results.json")["results"] if x["case"] == "video")
    b = [x for x in read("half-dense-conv-results.json")["workflows"] if x["case"] == "video"]
    rows = [("FP32", a["timings"]["layout"]["workflow_seconds"], a["timings"]["dense-conv"]["workflow_seconds"])]
    rows += [(x["precision"].upper(), x["timings"]["baseline"]["median"], x["timings"]["candidate"]["median"]) for x in b]
    fig, ax = plt.subplots(figsize=(13.5, 7))
    fig.subplots_adjust(left=.12, right=.77, bottom=.27, top=.75)
    header(fig, "Workflow time before and after optimization", "32 moving frames · M4 Pro, 48 GiB · median of three warmed runs",
           "Decode through uint16 rendering; PNG writing excluded. Separate before/after studies, not a cross-precision speed ranking.\n"
           "Exact raw and rendered agreement with each preceding precision path across 36 frames per precision. Zero added temporal alpha error.")
    for i, (name, old, new) in enumerate(rows):
        ax.barh(i + .2, old, height=.32, color="#78818f", label="Previous path" if i == 0 else None)
        ax.barh(i + .6, new, height=.32, color=COLORS[1], label="Enabled optimization" if i == 0 else None)
        ax.text(old + .5, i + .2, f"{old:.2f} s", va="center", fontsize=11)
        ax.text(new + .5, i + .6, f"{new:.2f} s", va="center", fontsize=11)
        ax.text(49.5, i + .35, f"{(1-new/old)*100:.1f}%", fontsize=24, color=COLORS[1], weight="bold", va="center")
        ax.text(49.5, i + .65, "less time", color=MUTED, fontsize=11, va="center")
    ax.set_yticks(np.arange(len(rows)) + .4, [x[0] for x in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 48); style(ax, "seconds / 32-frame workflow")
    fig.legend(*ax.get_legend_handles_labels(), loc="center", bbox_to_anchor=(.36, .125), ncol=2)
    save(fig, "optimization-gains")


def precision():
    d = read("q8-results.json")["cases"]
    labels = ["Green hair / gauze", "Green fur / motion", "Blue hair / gauze", "Blue fur / glass", "Moving clip (12 frames)"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.5))
    fig.subplots_adjust(left=.2, right=.95, bottom=.27, top=.75, wspace=.22)
    header(fig, "FP32, FP16 and Q8: workflow time and memory", "Matched three-way study · M4 Pro, 48 GiB · five warmed runs per case / precision",
           "Time includes four completed 16-bit PNG exports; video time is divided by 12 frames. Model load and warm-up excluded.\n"
           "Physical footprint is sampled with macOS proc_pid_rusage. FP16 and Q8 exceed strict FP32 numerical-agreement limits.")
    for j, variant in enumerate(("fp32", "fp16", "q8")):
        ys = np.arange(len(d)) + (j - 1) * .24
        seconds = [x["performance"][variant]["median_seconds"] / x["frames"] for x in d]
        memory = [x["performance"][variant]["peak_physical_gib"] for x in d]
        for ax, vals in zip(axes, (seconds, memory)):
            ax.barh(ys, vals, height=.2, color=COLORS[j], label=variant.upper())
    for ax, label, title in zip(axes, ("seconds / individual frame", "GiB · peak physical footprint"),
                                ("Warmed workflow time", "Physical memory")):
        ax.set_yticks(np.arange(len(d)), labels if ax is axes[0] else [""]*len(d))
        ax.invert_yaxis(); ax.set_title(title); style(ax, label)
        ax.set_xlim(left=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(.5, .125), ncol=3)
    save(fig, "precision-tradeoffs")


def kernel_latency():
    isolated = read("isolated-kernels.json")
    conv = read("channel-conv.json")
    groups = [
        ("Reconstruction", [(x["name"].replace("reconstruction-b", "Batch "), x["samples_seconds"]) for x in isolated if x["name"].startswith("reconstruction")]),
        ("Reflection padding", [(x["name"].replace("reflect-pad-c", "Channels ").replace("reflect-asymmetric", "Small asymmetric"), x["samples_seconds"]) for x in isolated if x["name"].startswith("reflect")]),
        ("Paired 7×7 convolution", [("Small control" if x["shape"][1] == 3 else f'Batch {x["shape"][0]} · 1280 channels', [x["samples"]["current"], x["samples"]["paired_conv"]]) for x in conv]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 8.2))
    fig.subplots_adjust(left=.13, right=.96, bottom=.27, top=.74, wspace=.7)
    header(fig, "Isolated FP32 kernel latency", "Apple M4 Pro, 48 GiB · compiled operators · bars: median · dots: individual samples",
           "Five alternating samples for reconstruction/padding; seven samples for convolution. Each panel has its own millisecond scale.\n"
           "Maximum absolute output difference: reconstruction 9.54e−6; padding and convolution 0. Kernel ratios are not whole-workflow speedups.")
    for ax, (title, rows) in zip(axes, groups):
        largest = max(max(samples) for _, pair in rows for samples in pair) * 1000
        for i, (_, pair) in enumerate(rows):
            for j, samples in enumerate(pair):
                values = np.array(samples) * 1000
                y = i + j * .3
                med = np.median(values)
                ax.barh(y, med, height=.25, color=[COLORS[0], COLORS[1]][j],
                        label=["Reference operator", "Optimized kernel"][j] if i == 0 else None)
                ax.scatter(values, np.full(len(values), y), s=12, color=INK, edgecolors=BG, linewidths=.3, zorder=4)
                ax.text(max(values) + largest*.035, y, f"{med:.2f}", va="center", fontsize=9)
        ax.set_yticks(np.arange(len(rows))+.15, [r[0] for r in rows], fontsize=10)
        ax.invert_yaxis(); ax.set_xlim(0, largest*1.33)
        ax.set_title(title); style(ax, "milliseconds / operator")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="center", bbox_to_anchor=(.5,.125), ncol=2)
    save(fig, "kernel-latency")


def kernel_ablation():
    data = read("half-operator-screen.json")
    names = ["Baseline", "Projection pairing", "Paired convolution", "Both changes"]
    modes = ["baseline", "half-paired", "half-conv", "half-dense-conv"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))
    fig.subplots_adjust(left=.19, right=.94, bottom=.27, top=.74, wspace=.18)
    header(fig, "Refinement-stage kernel ablation", "768 × 1280 · batch one · compiled stage 2 · two warm-ups, five measured runs with rotated order",
           "Synthetic tensor inputs, fixed random seed; dots show all measured runs. Every candidate matches its precision baseline exactly.\n"
           "Convolution alone is enabled for FP16/Q8. Projection pairing alone was slower, and adding it to convolution did not improve time.")
    for axis_index, (ax, precision) in enumerate(zip(axes, ("fp16", "q8"))):
        for i, mode in enumerate(modes):
            row = next(x for x in data if x["precision"] == precision and x["mode"] == mode)
            values = np.array(row["samples"]) * 1000
            ax.barh(i, row["median"]*1000, height=.55, color=COLORS[1] if mode=="half-conv" else COLORS[0])
            ax.scatter(values, np.full(len(values),i), s=18, color=INK, edgecolors=BG, linewidths=.4, zorder=4)
            ax.text(max(values)+9,i,f'{row["median"]*1000:.1f} ms',va="center",fontsize=10)
        ax.set_yticks(range(4), names if axis_index==0 else [""]*4)
        ax.invert_yaxis(); ax.set_xlim(0,780); ax.set_title(precision.upper())
        style(ax,"milliseconds / stage-two call")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COLORS[0],label="Control / other candidate"),Patch(color=COLORS[1],label="Enabled candidate")],
               loc="center",bbox_to_anchor=(.5,.125),ncol=2)
    save(fig,"kernel-ablation")


if __name__ == "__main__":
    baseline()
    optimization()
    precision()
    kernel_latency()
    kernel_ablation()
