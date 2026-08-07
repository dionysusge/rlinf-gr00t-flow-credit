#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = ["sft_base", "step400", "step600"]

EP_PATTERN = re.compile(
    r"\[libero eval\] "
    r"task_id=(\d+), "
    r"trial_id=(\d+), "
    r"success=(True|False)"
)


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=float), q))


def summary_stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p05": pct(x, 5),
        "p95": pct(x, 95),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def image_for_display(arr, batch_idx):
    if arr is None:
        return None

    x = arr[batch_idx]

    # [N_VIEW,H,W,C] -> first view
    if x.ndim == 4:
        x = x[0]

    if x.ndim != 3:
        return None

    if (
        x.shape[0] in (1, 3, 4)
        and x.shape[-1] not in (1, 3, 4)
    ):
        x = np.moveaxis(x, 0, -1)

    x = np.asarray(x)

    if np.issubdtype(x.dtype, np.floating):
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)

        if xmin < 0 or xmax > 1:
            if xmax > xmin:
                x = (x - xmin) / (xmax - xmin)

        x = np.clip(x, 0, 1)

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("repo_output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    aggregate = root / "aggregate"
    figures = aggregate / "figures"
    examples_dir = figures / "examples"

    aggregate.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.tsv"

    with manifest_path.open(encoding="utf-8") as f:
        manifest = list(
            csv.DictReader(f, delimiter="\t")
        )

    if len(manifest) != 15:
        raise RuntimeError(
            f"Expected 15 manifest rows, got {len(manifest)}"
        )

    # ============================================================
    # 1. Environment-level performance
    # ============================================================

    performance_rows = []

    for row in manifest:
        run = Path(row["output_dir"])

        performance_rows.append(
            {
                "label": row["label"],
                "offset": int(row["offset"]),
                "num_trajectories": int(
                    float(row["num_trajectories"])
                ),
                "success_once": float(row["success_once"]),
                "reward": float(row["reward"]),
                "return": float(row["return"]),
                "episode_len": float(row["episode_len"]),
                "run_rc": int(row["run_rc"]),
                "analysis_rc": int(row["analysis_rc"]),
                "raw_files": int(row["raw_files"]),
                "raw_bytes": int(row["raw_bytes"]),
                "output_dir": str(run),
            }
        )

    write_csv(
        aggregate / "subset_performance.csv",
        performance_rows,
    )

    model_performance = []

    for label in LABELS:
        rows = [
            r for r in performance_rows
            if r["label"] == label
        ]

        weights = np.asarray(
            [r["num_trajectories"] for r in rows],
            dtype=float,
        )

        def weighted(name):
            vals = np.asarray(
                [r[name] for r in rows],
                dtype=float,
            )
            return float(
                np.average(vals, weights=weights)
            )

        model_performance.append(
            {
                "label": label,
                "episodes": int(weights.sum()),
                "success_once": weighted("success_once"),
                "reward": weighted("reward"),
                "return": weighted("return"),
                "episode_len": weighted("episode_len"),
                "raw_files": sum(r["raw_files"] for r in rows),
                "raw_bytes": sum(r["raw_bytes"] for r in rows),
            }
        )

    write_csv(
        aggregate / "model_performance.csv",
        model_performance,
    )

    # ============================================================
    # 2. Parse per-episode outcomes from logs
    # ============================================================

    episode_dict = {}

    for row in manifest:
        label = row["label"]
        offset = int(row["offset"])
        log = Path(row["output_dir"]) / "evaluation.log"

        if not log.exists():
            continue

        for line in log.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            m = EP_PATTERN.search(line)

            if not m:
                continue

            task_id = int(m.group(1))
            trial_id = int(m.group(2))
            success = m.group(3) == "True"

            key = (
                label,
                offset,
                task_id,
                trial_id,
            )

            episode_dict[key] = {
                "label": label,
                "offset": offset,
                "task_id": task_id,
                "trial_id": trial_id,
                "success": int(success),
            }

    episode_rows = list(episode_dict.values())

    episode_rows.sort(
        key=lambda r: (
            LABELS.index(r["label"]),
            r["offset"],
            r["task_id"],
            r["trial_id"],
        )
    )

    write_csv(
        aggregate / "episode_outcomes.csv",
        episode_rows,
    )

    task_success_rows = []

    grouped = defaultdict(list)

    for row in episode_rows:
        grouped[
            (row["label"], row["task_id"])
        ].append(row["success"])

    for (label, task_id), values in sorted(
        grouped.items(),
        key=lambda x: (
            LABELS.index(x[0][0]),
            x[0][1],
        ),
    ):
        task_success_rows.append(
            {
                "label": label,
                "task_id": task_id,
                "episodes": len(values),
                "success_rate": float(np.mean(values)),
            }
        )

    write_csv(
        aggregate / "task_success_summary.csv",
        task_success_rows,
    )

    # ============================================================
    # 3. Expand all raw flow samples
    # ============================================================

    action_rows = []

    for manifest_row in manifest:
        label = manifest_row["label"]
        offset = int(manifest_row["offset"])
        run = Path(manifest_row["output_dir"])
        raw_dir = run / "raw"

        files = sorted(raw_dir.glob("flow_*.npz"))

        print(
            f"[LOAD] {label} offset={offset}: "
            f"{len(files)} raw files"
        )

        for path in files:
            data = np.load(
                path,
                allow_pickle=False,
            )

            endpoint_error = data["endpoint_error"]
            endpoint_distance = data["endpoint_distance"]
            alignment = data["alignment"]
            turning = data["turning_cos"]
            step_norm = data["step_norm"]

            path_ratio = data["path_ratio"]
            path_length = data["path_length"]
            chord_length = data["chord_length"]

            batch_size = len(path_ratio)

            if "task_descriptions" in data:
                tasks = data["task_descriptions"]
            else:
                tasks = np.asarray(
                    ["unknown"] * batch_size
                )

            call_idx = int(
                np.asarray(data["call_idx"]).reshape(-1)[0]
            )

            for b in range(batch_size):
                e = endpoint_error[b]
                d = endpoint_distance[b]
                a = alignment[b]
                t = turning[b]
                s = step_norm[b]

                record = {
                    "label": label,
                    "offset": offset,
                    "run_dir": str(run),
                    "raw_path": str(path),
                    "raw_file": path.name,
                    "batch_idx": b,
                    "call_idx": call_idx,
                    "task_description": str(
                        tasks[b]
                        if b < len(tasks)
                        else "unknown"
                    ),
                    "path_ratio": float(path_ratio[b]),
                    "path_excess": float(path_ratio[b] - 1.0),
                    "path_length": float(path_length[b]),
                    "chord_length": float(chord_length[b]),
                    "mean_endpoint_error": float(np.mean(e)),
                    "max_endpoint_error": float(np.max(e)),
                    "mean_alignment": float(np.mean(a)),
                    "min_alignment": float(np.min(a)),
                    "min_turning_cos": (
                        float(np.min(t))
                        if len(t)
                        else np.nan
                    ),
                    # Only e0->e1 and e1->e2 are non-trivial.
                    # e3 == 0 by Euler construction.
                    "endpoint_increase_01": int(
                        len(e) > 1 and e[1] > e[0]
                    ),
                    "endpoint_increase_12": int(
                        len(e) > 2 and e[2] > e[1]
                    ),
                }

                for k, value in enumerate(e):
                    record[f"endpoint_error_{k}"] = float(value)

                for k, value in enumerate(d):
                    record[f"endpoint_distance_{k}"] = float(value)

                for k, value in enumerate(a):
                    record[f"alignment_{k}"] = float(value)

                for k, value in enumerate(s):
                    record[f"step_norm_{k}"] = float(value)

                for k, value in enumerate(t):
                    record[f"turning_cos_{k}_{k+1}"] = float(value)

                action_rows.append(record)

    if not action_rows:
        raise RuntimeError("No action-level flow samples found")

    write_csv(
        aggregate / "action_chunk_metrics.csv",
        action_rows,
    )

    print(
        "[INFO] total action-chunk samples:",
        len(action_rows),
    )

    # ============================================================
    # 4. Model-level geometry summary
    # ============================================================

    geometry_rows = []

    for label in LABELS:
        rows = [
            r for r in action_rows
            if r["label"] == label
        ]

        out = {
            "label": label,
            "action_chunks": len(rows),
        }

        for metric in [
            "path_ratio",
            "path_excess",
            "path_length",
            "chord_length",
            "mean_endpoint_error",
            "max_endpoint_error",
            "mean_alignment",
            "min_alignment",
            "min_turning_cos",
        ]:
            stats = summary_stats(
                [r[metric] for r in rows]
            )

            for stat_name, value in stats.items():
                out[
                    f"{metric}_{stat_name}"
                ] = value

        out["endpoint_increase_01_rate"] = float(
            np.mean(
                [
                    r["endpoint_increase_01"]
                    for r in rows
                ]
            )
        )

        out["endpoint_increase_12_rate"] = float(
            np.mean(
                [
                    r["endpoint_increase_12"]
                    for r in rows
                ]
            )
        )

        for threshold in [1.02, 1.05, 1.10, 1.15]:
            out[
                f"path_ratio_gt_{threshold:.2f}_rate"
            ] = float(
                np.mean(
                    [
                        r["path_ratio"] > threshold
                        for r in rows
                    ]
                )
            )

        for threshold in [0.9, 0.75, 0.5, 0.0]:
            out[
                f"min_turning_lt_{threshold:.2f}_rate"
            ] = float(
                np.mean(
                    [
                        r["min_turning_cos"] < threshold
                        for r in rows
                        if np.isfinite(
                            r["min_turning_cos"]
                        )
                    ]
                )
            )

        geometry_rows.append(out)

    write_csv(
        aggregate / "model_geometry_summary.csv",
        geometry_rows,
    )

    # ============================================================
    # 5. Per-flow-step statistics
    # ============================================================

    step_rows = []

    for label in LABELS:
        rows = [
            r for r in action_rows
            if r["label"] == label
        ]

        for k in range(4):
            for metric in [
                "endpoint_error",
                "endpoint_distance",
                "alignment",
                "step_norm",
            ]:
                values = [
                    r[f"{metric}_{k}"]
                    for r in rows
                ]

                stats = summary_stats(values)

                step_rows.append(
                    {
                        "label": label,
                        "kind": metric,
                        "step": k,
                        **stats,
                    }
                )

        for k in range(3):
            metric = f"turning_cos_{k}_{k+1}"

            stats = summary_stats(
                [r[metric] for r in rows]
            )

            step_rows.append(
                {
                    "label": label,
                    "kind": "turning_cos",
                    "step": k,
                    **stats,
                }
            )

    write_csv(
        aggregate / "flow_step_summary.csv",
        step_rows,
    )

    # ============================================================
    # 6. Geometry grouped by task text
    # ============================================================

    task_geometry_groups = defaultdict(list)

    for r in action_rows:
        task_geometry_groups[
            (
                r["label"],
                r["task_description"],
            )
        ].append(r)

    task_geometry_rows = []

    for (
        label,
        task_description,
    ), rows in sorted(
        task_geometry_groups.items(),
        key=lambda x: (
            LABELS.index(x[0][0]),
            x[0][1],
        ),
    ):
        task_geometry_rows.append(
            {
                "label": label,
                "task_description": task_description,
                "action_chunks": len(rows),
                "path_ratio_mean": float(
                    np.mean(
                        [r["path_ratio"] for r in rows]
                    )
                ),
                "path_ratio_median": float(
                    np.median(
                        [r["path_ratio"] for r in rows]
                    )
                ),
                "mean_alignment": float(
                    np.mean(
                        [r["mean_alignment"] for r in rows]
                    )
                ),
                "min_alignment_mean": float(
                    np.mean(
                        [r["min_alignment"] for r in rows]
                    )
                ),
                "min_turning_cos_mean": float(
                    np.mean(
                        [r["min_turning_cos"] for r in rows]
                    )
                ),
                "endpoint_increase_01_rate": float(
                    np.mean(
                        [
                            r["endpoint_increase_01"]
                            for r in rows
                        ]
                    )
                ),
                "endpoint_increase_12_rate": float(
                    np.mean(
                        [
                            r["endpoint_increase_12"]
                            for r in rows
                        ]
                    )
                ),
            }
        )

    write_csv(
        aggregate / "task_geometry_summary.csv",
        task_geometry_rows,
    )

    # ============================================================
    # 7. Figures
    # ============================================================

    # Success rate
    fig, ax = plt.subplots(figsize=(7, 4.5))

    perf_map = {
        r["label"]: r
        for r in model_performance
    }

    ax.bar(
        LABELS,
        [
            100 * perf_map[l]["success_once"]
            for l in LABELS
        ],
    )

    ax.set_ylabel("Success rate (%)")
    ax.set_title("LIBERO-Spatial fixed500 performance")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        figures / "checkpoint_success_rate.png",
        dpi=200,
    )
    plt.close(fig)

    # Path ratio distribution
    fig, ax = plt.subplots(figsize=(8, 5))

    for label in LABELS:
        values = [
            r["path_ratio"]
            for r in action_rows
            if r["label"] == label
        ]

        ax.hist(
            values,
            bins=60,
            histtype="step",
            density=True,
            label=label,
        )

    ax.axvline(1.0, linestyle="--")
    ax.set_xlabel("Path length / chord length")
    ax.set_ylabel("Density")
    ax.set_title("Flow path-ratio distribution")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        figures / "path_ratio_distribution.png",
        dpi=200,
    )
    plt.close(fig)

    # Path ratio boxplot
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.boxplot(
        [
            [
                r["path_ratio"]
                for r in action_rows
                if r["label"] == label
            ]
            for label in LABELS
        ],
        tick_labels=LABELS,
        showfliers=False,
    )

    ax.axhline(1.0, linestyle="--")
    ax.set_ylabel("Path ratio")
    ax.set_title("Flow path straightness by checkpoint")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures / "path_ratio_boxplot.png",
        dpi=200,
    )
    plt.close(fig)

    def plot_step_metric(
        kind,
        ylabel,
        title,
        filename,
        n_steps,
    ):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        for label in LABELS:
            means = []
            p05 = []
            p95 = []

            for k in range(n_steps):
                match = next(
                    r
                    for r in step_rows
                    if (
                        r["label"] == label
                        and r["kind"] == kind
                        and r["step"] == k
                    )
                )

                means.append(match["mean"])
                p05.append(match["p05"])
                p95.append(match["p95"])

            x = np.arange(n_steps)

            line = ax.plot(
                x,
                means,
                marker="o",
                label=label,
            )[0]

            ax.fill_between(
                x,
                p05,
                p95,
                alpha=0.10,
            )

        ax.set_xticks(np.arange(n_steps))
        ax.set_xlabel(
            "Flow step"
            if kind != "turning_cos"
            else "Transition start step"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            figures / filename,
            dpi=200,
        )
        plt.close(fig)

    plot_step_metric(
        "endpoint_error",
        "L2 error",
        "Endpoint extrapolation error",
        "endpoint_error_by_step.png",
        4,
    )

    plot_step_metric(
        "endpoint_distance",
        "L2 distance",
        "Distance from current flow state to final endpoint",
        "endpoint_distance_by_step.png",
        4,
    )

    plot_step_metric(
        "alignment",
        "Cosine similarity",
        "Velocity-to-endpoint alignment",
        "endpoint_alignment_by_step.png",
        4,
    )

    plot_step_metric(
        "step_norm",
        "L2 displacement",
        "Flow-step displacement",
        "step_norm_by_step.png",
        4,
    )

    plot_step_metric(
        "turning_cos",
        "Cosine similarity",
        "Consecutive velocity turning",
        "turning_cosine_by_transition.png",
        3,
    )

    # Non-monotonic endpoint extrapolation rates
    fig, ax = plt.subplots(figsize=(8, 4.8))

    x = np.arange(len(LABELS))
    width = 0.34

    rate01 = []
    rate12 = []

    geom_map = {
        r["label"]: r
        for r in geometry_rows
    }

    for label in LABELS:
        rate01.append(
            100
            * geom_map[label][
                "endpoint_increase_01_rate"
            ]
        )
        rate12.append(
            100
            * geom_map[label][
                "endpoint_increase_12_rate"
            ]
        )

    ax.bar(
        x - width / 2,
        rate01,
        width,
        label="e1 > e0",
    )

    ax.bar(
        x + width / 2,
        rate12,
        width,
        label="e2 > e1",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Action chunks (%)")
    ax.set_title(
        "Non-monotonic endpoint extrapolation"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figures / "endpoint_nonmonotonic_rate.png",
        dpi=200,
    )
    plt.close(fig)

    # ============================================================
    # 8. Representative extreme examples
    # ============================================================

    selected = []

    for label in LABELS:
        rows = [
            r for r in action_rows
            if r["label"] == label
        ]

        highest_ratio = sorted(
            rows,
            key=lambda r: r["path_ratio"],
            reverse=True,
        )[:2]

        lowest_turn = sorted(
            rows,
            key=lambda r: r["min_turning_cos"],
        )[:2]

        candidates = highest_ratio + lowest_turn

        seen = set()

        for r in candidates:
            key = (
                r["raw_path"],
                r["batch_idx"],
            )

            if key not in seen:
                selected.append(r)
                seen.add(key)

    for idx, r in enumerate(selected):
        data = np.load(
            r["raw_path"],
            allow_pickle=False,
        )

        b = int(r["batch_idx"])

        cameras = []

        for title, key in [
            ("Main camera", "main_images"),
            ("Wrist camera", "wrist_images"),
            ("Extra camera", "extra_view_images"),
        ]:
            img = (
                image_for_display(data[key], b)
                if key in data
                else None
            )
            cameras.append((title, img))

        e = data["endpoint_error"][b]
        a = data["alignment"][b]
        t = data["turning_cos"][b]
        s = data["step_norm"][b]

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3)

        for j, (title, img) in enumerate(cameras):
            ax = fig.add_subplot(gs[0, j])
            ax.set_title(title)
            ax.axis("off")

            if img is None:
                ax.text(
                    0.5,
                    0.5,
                    "not available",
                    ha="center",
                    va="center",
                )
            else:
                ax.imshow(img)

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(
            np.arange(len(e)),
            e,
            marker="o",
        )
        ax.set_title("Endpoint extrapolation error")
        ax.set_xlabel("Flow step")
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(
            np.arange(len(a)),
            a,
            marker="o",
        )
        ax.axhline(0, linestyle="--")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title("Endpoint alignment")
        ax.set_xlabel("Flow step")
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(
            np.arange(len(s)),
            s,
            marker="o",
            label="step norm",
        )

        ax2 = ax.twinx()
        ax2.plot(
            np.arange(len(t)),
            t,
            marker="x",
            linestyle="--",
            label="turning cosine",
        )
        ax2.set_ylim(-1.05, 1.05)

        ax.set_title(
            f"path ratio={r['path_ratio']:.3f}, "
            f"min turn={r['min_turning_cos']:.3f}"
        )
        ax.set_xlabel("Flow step")
        ax.grid(alpha=0.25)

        fig.suptitle(
            f"{r['label']} | {r['task_description']}",
            fontsize=11,
        )

        fig.tight_layout()

        fig.savefig(
            examples_dir
            / (
                f"{idx:02d}_"
                f"{r['label']}_"
                f"ratio{r['path_ratio']:.3f}_"
                f"turn{r['min_turning_cos']:.3f}.png"
            ),
            dpi=180,
        )

        plt.close(fig)

    # ============================================================
    # 9. Markdown report
    # ============================================================

    perf_map = {
        r["label"]: r
        for r in model_performance
    }

    geom_map = {
        r["label"]: r
        for r in geometry_rows
    }

    md = []

    md.append("# GR00T N1.7 Flow Geometry Fixed500 × 3\n")

    md.append(
        "> Geometry metrics are descriptive diagnostics. "
        "They do not by themselves identify a harmful flow step.\n"
    )

    md.append("## Environment performance\n")

    md.append(
        "| Model | Episodes | Success | Reward | Episode length |\n"
        "|---|---:|---:|---:|---:|"
    )

    for label in LABELS:
        p = perf_map[label]

        md.append(
            f"| {label} | {p['episodes']} | "
            f"{100*p['success_once']:.2f}% | "
            f"{p['reward']:.6f} | "
            f"{p['episode_len']:.2f} |"
        )

    md.append("\n## Flow geometry\n")

    md.append(
        "| Model | Action chunks | Path ratio mean | "
        "Path ratio median | Alignment mean | "
        "Min-turn mean | e1>e0 | e2>e1 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    for label in LABELS:
        g = geom_map[label]

        md.append(
            f"| {label} | {g['action_chunks']} | "
            f"{g['path_ratio_mean']:.4f} | "
            f"{g['path_ratio_median']:.4f} | "
            f"{g['mean_alignment_mean']:.4f} | "
            f"{g['min_turning_cos_mean']:.4f} | "
            f"{100*g['endpoint_increase_01_rate']:.2f}% | "
            f"{100*g['endpoint_increase_12_rate']:.2f}% |"
        )

    md.append(
        "\n## Interpretation constraints\n\n"
        "- `endpoint_error_3 = 0` is guaranteed by the final Euler step; "
        "it is not an empirical finding.\n"
        "- `e1 > e0` and `e2 > e1` indicate non-monotonic endpoint "
        "extrapolation, not automatically a wrong transition.\n"
        "- Raw flow dumps currently do not carry an explicit episode-success "
        "identifier, so action-level success/failure geometry is intentionally "
        "not reported here.\n"
        "- `episode_outcomes.csv` contains exact task/trial success outcomes "
        "and can be used after sample-to-episode mapping is verified.\n"
    )

    report = aggregate / "REPORT.md"
    report.write_text(
        "\n".join(md),
        encoding="utf-8",
    )

    # ============================================================
    # 10. Copy curated results into Git repository
    # ============================================================

    repo_output = args.repo_output.resolve()

    if repo_output.exists():
        shutil.rmtree(repo_output)

    repo_output.mkdir(parents=True)

    for filename in [
        "REPORT.md",
        "model_performance.csv",
        "model_geometry_summary.csv",
        "flow_step_summary.csv",
        "task_success_summary.csv",
        "task_geometry_summary.csv",
        "subset_performance.csv",
    ]:
        shutil.copy2(
            aggregate / filename,
            repo_output / filename,
        )

    shutil.copytree(
        figures,
        repo_output / "figures",
    )

    print()
    print("========================================")
    print("AGGREGATE SUMMARY")
    print("========================================")

    print("episodes:", sum(
        r["episodes"] for r in model_performance
    ))

    print("action chunks:", len(action_rows))

    print()

    for label in LABELS:
        p = perf_map[label]
        g = geom_map[label]

        print(
            label,
            f"success={p['success_once']:.4f}",
            f"path_ratio={g['path_ratio_mean']:.4f}",
            f"alignment={g['mean_alignment_mean']:.4f}",
            f"turn={g['min_turning_cos_mean']:.4f}",
            f"e01_up={g['endpoint_increase_01_rate']:.4f}",
            f"e12_up={g['endpoint_increase_12_rate']:.4f}",
        )

    print()
    print("aggregate:", aggregate)
    print("repo output:", repo_output)


if __name__ == "__main__":
    main()
