#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate per-step residual dumps into research-ready CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DIMENSION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
PID_PATTERN = re.compile(r"pid(\d+)")


def read_outcomes(path: Path | None) -> dict[tuple[int, int], int]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["task_id"]), int(row["trial_id"])): int(row["success"]) for row in rows
    }


def scalar_rows(
    diagnostic_dir: Path,
    outcomes: dict[tuple[int, int], int],
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows = []
    chunks = []
    for path in sorted(diagnostic_dir.glob("residual_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            residual = np.asarray(payload["residual_action"], dtype=np.float64)
            base = np.asarray(payload["base_action"], dtype=np.float64)
            task_ids = np.asarray(
                payload.get("task_ids", np.full(residual.shape[0], -1))
            ).reshape(-1)
            trial_ids = np.asarray(
                payload.get("trial_ids", np.full(residual.shape[0], -1))
            ).reshape(-1)
            reset_ids = np.asarray(
                payload.get("reset_ids", np.full(residual.shape[0], -1))
            ).reshape(-1)
            call_idx = int(np.asarray(payload["call_idx"]).item())
            match = PID_PATTERN.search(path.name)
            worker_pid = int(match.group(1)) if match else -1

        chunks.append(residual)
        step_norm = np.linalg.norm(residual, axis=-1)
        base_norm = np.linalg.norm(base, axis=-1)
        for batch_idx in range(residual.shape[0]):
            task_id = int(task_ids[batch_idx])
            trial_id = int(trial_ids[batch_idx])
            item = residual[batch_idx]
            rows.append(
                {
                    "worker_pid": worker_pid,
                    "call_idx": call_idx,
                    "batch_idx": batch_idx,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "reset_id": int(reset_ids[batch_idx]),
                    "success": outcomes.get((task_id, trial_id), ""),
                    "residual_l2_mean": float(step_norm[batch_idx].mean()),
                    "residual_l2_max": float(step_norm[batch_idx].max()),
                    "residual_abs_mean": float(np.abs(item).mean()),
                    "translation_abs_mean": float(np.abs(item[..., :3]).mean()),
                    "rotation_abs_mean": float(np.abs(item[..., 3:6]).mean()),
                    "gripper_abs_mean": float(np.abs(item[..., 6:7]).mean())
                    if item.shape[-1] >= 7
                    else float("nan"),
                    "residual_to_base_l2_ratio": float(
                        (step_norm[batch_idx] / (base_norm[batch_idx] + 1e-8)).mean()
                    ),
                }
            )
    if not chunks:
        raise ValueError(f"No residual_*.npz files found in {diagnostic_dir}")
    return rows, np.concatenate(chunks, axis=0)


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--trials-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_outcomes(args.trials_csv)
    rows, residual = scalar_rows(args.diagnostic_dir, outcomes)
    write_csv(args.output_dir / "step_records.csv", rows)

    horizon_norm = np.linalg.norm(residual, axis=-1)
    horizon_rows = []
    for horizon in range(horizon_norm.shape[1]):
        horizon_rows.append({"horizon": horizon, **describe(horizon_norm[:, horizon])})
    write_csv(args.output_dir / "per_horizon.csv", horizon_rows)

    dimension_rows = []
    for dimension in range(residual.shape[-1]):
        name = (
            DIMENSION_NAMES[dimension]
            if dimension < len(DIMENSION_NAMES)
            else f"dim_{dimension}"
        )
        dimension_rows.append(
            {
                "dimension": dimension,
                "name": name,
                **describe(np.abs(residual[..., dimension])),
            }
        )
    write_csv(args.output_dir / "per_dimension.csv", dimension_rows)

    conditioned = defaultdict(list)
    for row in rows:
        if row["success"] != "":
            conditioned[int(row["success"])].append(float(row["residual_l2_mean"]))
    conditioned_rows = [
        {"success": success, **describe(np.asarray(values))}
        for success, values in sorted(conditioned.items())
    ]
    if conditioned_rows:
        write_csv(args.output_dir / "success_conditioned.csv", conditioned_rows)

    summary = {
        "num_diagnostic_samples": int(residual.shape[0]),
        "action_horizon": int(residual.shape[1]),
        "action_dim": int(residual.shape[2]),
        "residual_l2": describe(horizon_norm),
        "residual_abs": describe(np.abs(residual)),
        "active_fraction_gt_0.01": float((horizon_norm > 0.01).mean()),
        "active_fraction_gt_0.05": float((horizon_norm > 0.05).mean()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
