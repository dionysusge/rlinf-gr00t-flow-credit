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
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(diagnostic_dir.glob("residual_*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            residual = np.asarray(payload["residual_action"], dtype=np.float64)
            base = np.asarray(payload["base_action"], dtype=np.float64)
            residual_bound = float(
                np.asarray(payload.get("residual_bound", 0.1)).item()
            )
            eval_scale = float(np.asarray(payload.get("eval_scale", 1.0)).item())
            if "residual_mean_action" in payload:
                mean_action = np.asarray(
                    payload["residual_mean_action"], dtype=np.float64
                )
            else:
                mean_action = (
                    residual_bound
                    * eval_scale
                    * np.tanh(np.asarray(payload["residual_mean"], dtype=np.float64))
                )
            exploration_action = np.asarray(
                payload.get("residual_exploration_action", residual - mean_action),
                dtype=np.float64,
            )
            log_std = np.asarray(payload["residual_log_std"], dtype=np.float64)
            executed = np.asarray(
                payload.get("executed_normalized_action", base + residual),
                dtype=np.float64,
            )
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

        arrays["sample"].append(residual)
        arrays["mean"].append(mean_action)
        arrays["exploration"].append(exploration_action)
        arrays["log_std"].append(log_std)
        arrays["base"].append(base)
        arrays["executed"].append(executed)
        step_norm = np.linalg.norm(residual, axis=-1)
        mean_norm = np.linalg.norm(mean_action, axis=-1)
        exploration_norm = np.linalg.norm(exploration_action, axis=-1)
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
                    "sample_l2_mean": float(step_norm[batch_idx].mean()),
                    "sample_l2_max": float(step_norm[batch_idx].max()),
                    "mean_l2_mean": float(mean_norm[batch_idx].mean()),
                    "exploration_l2_mean": float(exploration_norm[batch_idx].mean()),
                    "sample_abs_mean": float(np.abs(item).mean()),
                    "mean_abs_mean": float(np.abs(mean_action[batch_idx]).mean()),
                    "exploration_abs_mean": float(
                        np.abs(exploration_action[batch_idx]).mean()
                    ),
                    "sample_saturation_fraction_gt_0.09": float(
                        (np.abs(item) > 0.09).mean()
                    ),
                    "translation_abs_mean": float(np.abs(item[..., :3]).mean()),
                    "rotation_abs_mean": float(np.abs(item[..., 3:6]).mean())
                    if item.shape[-1] >= 6
                    else float("nan"),
                    "gripper_abs_mean": float(np.abs(item[..., 6:7]).mean())
                    if item.shape[-1] >= 7
                    else float("nan"),
                    "residual_to_base_l2_ratio": float(
                        (step_norm[batch_idx] / (base_norm[batch_idx] + 1e-8)).mean()
                    ),
                    "base_normalized_ood_fraction": float(
                        (np.abs(base[batch_idx]) > 1.0).mean()
                    ),
                    "executed_normalized_ood_fraction": float(
                        (np.abs(executed[batch_idx]) > 1.0).mean()
                    ),
                    "created_normalized_ood_fraction": float(
                        (
                            (np.abs(base[batch_idx]) <= 1.0)
                            & (np.abs(executed[batch_idx]) > 1.0)
                        ).mean()
                    ),
                }
            )
    if not arrays:
        raise ValueError(f"No residual_*.npz files found in {diagnostic_dir}")
    return rows, {
        name: np.concatenate(chunks, axis=0) for name, chunks in arrays.items()
    }


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
    rows, arrays = scalar_rows(args.diagnostic_dir, outcomes)
    write_csv(args.output_dir / "step_records.csv", rows)

    residual = arrays["sample"]
    mean_action = arrays["mean"]
    exploration_action = arrays["exploration"]
    log_std = arrays["log_std"]
    base_action = arrays["base"]
    executed_action = arrays["executed"]
    horizon_norm = np.linalg.norm(residual, axis=-1)
    mean_horizon_norm = np.linalg.norm(mean_action, axis=-1)
    exploration_horizon_norm = np.linalg.norm(exploration_action, axis=-1)
    horizon_rows = []
    for horizon in range(horizon_norm.shape[1]):
        horizon_rows.append(
            {
                "horizon": horizon,
                **{
                    f"sample_l2_{k}": v
                    for k, v in describe(horizon_norm[:, horizon]).items()
                },
                **{
                    f"mean_l2_{k}": v
                    for k, v in describe(mean_horizon_norm[:, horizon]).items()
                },
                **{
                    f"exploration_l2_{k}": v
                    for k, v in describe(exploration_horizon_norm[:, horizon]).items()
                },
                "sample_saturation_fraction_gt_0.09": float(
                    (np.abs(residual[:, horizon]) > 0.09).mean()
                ),
            }
        )
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
                **{
                    f"sample_abs_{k}": v
                    for k, v in describe(np.abs(residual[..., dimension])).items()
                },
                **{
                    f"mean_abs_{k}": v
                    for k, v in describe(np.abs(mean_action[..., dimension])).items()
                },
                "sample_saturation_fraction_gt_0.09": float(
                    (np.abs(residual[..., dimension]) > 0.09).mean()
                ),
                "log_std_mean": float(log_std[..., dimension].mean()),
                "std_mean": float(np.exp(log_std[..., dimension]).mean()),
            }
        )
    write_csv(args.output_dir / "per_dimension.csv", dimension_rows)

    conditioned = defaultdict(list)
    for row in rows:
        if row["success"] != "":
            conditioned[int(row["success"])].append(row)
    conditioned_rows = [
        {
            "success": success,
            **{
                f"sample_l2_{k}": v
                for k, v in describe(
                    np.asarray([row["sample_l2_mean"] for row in values])
                ).items()
            },
            **{
                f"mean_l2_{k}": v
                for k, v in describe(
                    np.asarray([row["mean_l2_mean"] for row in values])
                ).items()
            },
            **{
                f"exploration_l2_{k}": v
                for k, v in describe(
                    np.asarray([row["exploration_l2_mean"] for row in values])
                ).items()
            },
        }
        for success, values in sorted(conditioned.items())
    ]
    if conditioned_rows:
        write_csv(args.output_dir / "success_conditioned.csv", conditioned_rows)

    summary = {
        "num_diagnostic_samples": int(residual.shape[0]),
        "action_horizon": int(residual.shape[1]),
        "action_dim": int(residual.shape[2]),
        "sample_residual_l2": describe(horizon_norm),
        "mean_residual_l2": describe(mean_horizon_norm),
        "exploration_residual_l2": describe(exploration_horizon_norm),
        "sample_residual_abs": describe(np.abs(residual)),
        "log_std": describe(log_std),
        "std": describe(np.exp(log_std)),
        "sample_active_fraction_gt_0.01": float((horizon_norm > 0.01).mean()),
        "sample_active_fraction_gt_0.05": float((horizon_norm > 0.05).mean()),
        "sample_saturation_fraction_gt_0.09": float((np.abs(residual) > 0.09).mean()),
        "base_normalized_ood_fraction": float((np.abs(base_action) > 1.0).mean()),
        "executed_normalized_ood_fraction": float(
            (np.abs(executed_action) > 1.0).mean()
        ),
        "created_normalized_ood_fraction": float(
            ((np.abs(base_action) <= 1.0) & (np.abs(executed_action) > 1.0)).mean()
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
