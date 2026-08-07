#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def image_for_display(arr, batch_idx):
    if arr is None:
        return None

    x = arr

    if x.ndim >= 4:
        x = x[batch_idx]

    # Extra-view format may be [N_VIEW,H,W,C]
    if x.ndim == 4:
        x = x[0]

    if x.ndim != 3:
        return None

    # CHW -> HWC
    if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.moveaxis(x, 0, -1)

    if np.issubdtype(x.dtype, np.floating):
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        if xmin < 0 or xmax > 1:
            if xmax > xmin:
                x = (x - xmin) / (xmax - xmin)
        x = np.clip(x, 0, 1)

    return x


def plot_mean_curve(values, title, ylabel, output):
    stack = np.concatenate(values, axis=0)
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    x = np.arange(stack.shape[1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, mean, marker="o")
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Flow step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("fig_dir", type=Path)
    args = parser.parse_args()

    files = sorted(args.raw_dir.glob("flow_*.npz"))
    if not files:
        raise SystemExit(f"No diagnostic files in {args.raw_dir}")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.fig_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    endpoint_errors = []
    alignments = []
    turnings = []
    path_ratios = []

    rows = []
    examples = []

    for path in files:
        data = np.load(path, allow_pickle=False)

        ee = data["endpoint_error"]
        al = data["alignment"]
        tc = data["turning_cos"]
        pr = data["path_ratio"]
        sn = data["step_norm"]

        endpoint_errors.append(ee)
        alignments.append(al)

        if tc.shape[1] > 0:
            turnings.append(tc)

        path_ratios.append(pr)

        batch_size = ee.shape[0]

        task_desc = (
            data["task_descriptions"]
            if "task_descriptions" in data
            else np.asarray(["unknown"] * batch_size)
        )

        for b in range(batch_size):
            row = {
                "file": path.name,
                "batch_idx": b,
                "task": str(
                    task_desc[b]
                    if b < len(task_desc)
                    else "unknown"
                ),
                "path_ratio": float(pr[b]),
                "path_length": float(data["path_length"][b]),
                "chord_length": float(data["chord_length"][b]),
                "mean_endpoint_error": float(np.mean(ee[b])),
                "max_endpoint_error": float(np.max(ee[b])),
                "mean_alignment": float(np.mean(al[b])),
                "min_alignment": float(np.min(al[b])),
                "min_turning_cos": (
                    float(np.min(tc[b]))
                    if tc.shape[1] > 0
                    else float("nan")
                ),
            }

            rows.append(row)

            examples.append(
                {
                    "path": path,
                    "batch_idx": b,
                    "row": row,
                    "endpoint_error": ee[b],
                    "alignment": al[b],
                    "turning": tc[b],
                    "step_norm": sn[b],
                }
            )

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    csv_path = args.fig_dir / "geometry_summary.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Mean curves
    # --------------------------------------------------------

    plot_mean_curve(
        endpoint_errors,
        "Endpoint extrapolation error",
        "L2 error",
        args.fig_dir / "endpoint_error_mean.png",
    )

    plot_mean_curve(
        alignments,
        "Velocity-to-endpoint alignment",
        "Cosine similarity",
        args.fig_dir / "endpoint_alignment_mean.png",
    )

    if turnings:
        plot_mean_curve(
            turnings,
            "Consecutive velocity turning",
            "cos(v_k, v_{k+1})",
            args.fig_dir / "turning_cosine_mean.png",
        )

    # --------------------------------------------------------
    # Path ratio histogram
    # --------------------------------------------------------

    all_ratios = np.concatenate(path_ratios)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(all_ratios, bins=min(30, max(5, len(all_ratios))))
    ax.axvline(1.0, linestyle="--")
    ax.set_xlabel("Path length / endpoint chord length")
    ax.set_ylabel("Count")
    ax.set_title("Flow path straightness")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(
        args.fig_dir / "path_ratio_histogram.png",
        dpi=180,
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Overall metric summary
    # --------------------------------------------------------

    all_ee = np.concatenate(endpoint_errors, axis=0)
    all_al = np.concatenate(alignments, axis=0)

    print("========================================")
    print("FLOW GEOMETRY SUMMARY")
    print("========================================")
    print("files:", len(files))
    print("samples:", len(rows))
    print(
        "path ratio mean / median / max:",
        float(np.mean(all_ratios)),
        float(np.median(all_ratios)),
        float(np.max(all_ratios)),
    )
    print(
        "alignment mean / min:",
        float(np.mean(all_al)),
        float(np.min(all_al)),
    )
    print(
        "endpoint error per step mean:",
        np.mean(all_ee, axis=0).tolist(),
    )

    if turnings:
        all_tc = np.concatenate(turnings, axis=0)
        print(
            "turning cosine mean per transition:",
            np.mean(all_tc, axis=0).tolist(),
        )
        print(
            "turning cosine global min:",
            float(np.min(all_tc)),
        )

    # --------------------------------------------------------
    # Example contact sheets
    #
    # Highest path-ratio examples are useful for inspecting
    # strongly curved flow trajectories.
    # --------------------------------------------------------

    examples.sort(
        key=lambda x: x["row"]["path_ratio"],
        reverse=True,
    )

    for rank, ex in enumerate(examples[:12]):
        data = np.load(ex["path"], allow_pickle=False)
        b = ex["batch_idx"]

        main_img = image_for_display(
            data["main_images"]
            if "main_images" in data
            else None,
            b,
        )

        wrist_img = image_for_display(
            data["wrist_images"]
            if "wrist_images" in data
            else None,
            b,
        )

        extra_img = image_for_display(
            data["extra_view_images"]
            if "extra_view_images" in data
            else None,
            b,
        )

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3)

        cameras = [
            ("Main camera", main_img),
            ("Wrist camera", wrist_img),
            ("Extra camera", extra_img),
        ]

        for i, (name, image) in enumerate(cameras):
            ax = fig.add_subplot(gs[0, i])
            ax.set_title(name)
            ax.axis("off")

            if image is not None:
                ax.imshow(image)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "not available",
                    ha="center",
                    va="center",
                )

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(
            np.arange(len(ex["endpoint_error"])),
            ex["endpoint_error"],
            marker="o",
        )
        ax.set_title("Endpoint extrapolation error")
        ax.set_xlabel("Flow step")
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(
            np.arange(len(ex["alignment"])),
            ex["alignment"],
            marker="o",
            label="endpoint alignment",
        )
        ax.axhline(0, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Velocity alignment")
        ax.set_xlabel("Flow step")
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[1, 2])

        x1 = np.arange(len(ex["step_norm"]))
        ax.plot(
            x1,
            ex["step_norm"],
            marker="o",
            label="step displacement",
        )

        if len(ex["turning"]):
            x2 = np.arange(len(ex["turning"]))
            ax2 = ax.twinx()
            ax2.plot(
                x2,
                ex["turning"],
                marker="x",
                linestyle="--",
                label="turning cosine",
            )
            ax2.set_ylim(-1.05, 1.05)
            ax2.set_ylabel("Turning cosine")

        ax.set_title(
            "Local motion\n"
            f"path ratio={ex['row']['path_ratio']:.3f}"
        )
        ax.set_xlabel("Flow step")
        ax.set_ylabel("Step displacement")
        ax.grid(alpha=0.25)

        fig.suptitle(
            ex["row"]["task"],
            fontsize=11,
        )

        fig.tight_layout()

        output = examples_dir / (
            f"rank{rank:02d}_"
            f"pathratio{ex['row']['path_ratio']:.3f}.png"
        )

        fig.savefig(output, dpi=170)
        plt.close(fig)

    print("figures:", args.fig_dir)


if __name__ == "__main__":
    main()
