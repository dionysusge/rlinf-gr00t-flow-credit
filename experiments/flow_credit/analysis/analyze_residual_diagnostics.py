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
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DIMENSION_NAMES = ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
PID_PATTERN = re.compile(r"pid(\d+)")
SATURATION_THRESHOLD = 0.09
RAW_PRESSURE_THRESHOLD = math.atanh(0.9)
PROGRESS_BIN_NAMES = ("early", "middle", "late")


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.max()) if finite.size else float("nan")


def read_outcomes(path: Path | None) -> dict[tuple[int, int], int]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["task_id"]), int(row["trial_id"])): int(row["success"]) for row in rows
    }


def read_pairing(path: Path | None) -> dict[tuple[int, int], str]:
    """Read preserve/rescue/harm/unresolved labels from paired evaluation."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["task_id"]), int(row["trial_id"])): row["transition"] for row in rows
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
            raw_action = np.asarray(
                payload.get(
                    "residual_raw_action",
                    np.arctanh(
                        np.clip(
                            residual / max(residual_bound * eval_scale, 1e-8),
                            -1 + 1e-7,
                            1 - 1e-7,
                        )
                    ),
                ),
                dtype=np.float64,
            )
            raw_mean = np.asarray(
                payload.get(
                    "residual_mean",
                    np.arctanh(
                        np.clip(
                            mean_action / max(residual_bound * eval_scale, 1e-8),
                            -1 + 1e-7,
                            1 - 1e-7,
                        )
                    ),
                ),
                dtype=np.float64,
            )
            executed = np.asarray(
                payload.get("executed_normalized_action", base + residual),
                dtype=np.float64,
            )
            environment_base = np.asarray(
                payload.get("environment_base_action", np.full_like(base, np.nan)),
                dtype=np.float64,
            )
            environment_executed = np.asarray(
                payload.get(
                    "environment_executed_action",
                    np.full_like(executed, np.nan),
                ),
                dtype=np.float64,
            )
            environment_residual = np.asarray(
                payload.get(
                    "environment_residual_action",
                    environment_executed - environment_base,
                ),
                dtype=np.float64,
            )
            decoded_base = np.asarray(
                payload.get("decoded_base_action", environment_base),
                dtype=np.float64,
            )
            decoded_executed = np.asarray(
                payload.get("decoded_executed_action", environment_executed),
                dtype=np.float64,
            )
            decoded_residual = np.asarray(
                payload.get(
                    "decoded_residual_action",
                    decoded_executed - decoded_base,
                ),
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
        arrays["raw_sample"].append(raw_action)
        arrays["raw_mean"].append(raw_mean)
        arrays["log_std"].append(log_std)
        arrays["base"].append(base)
        arrays["executed"].append(executed)
        arrays["environment_base"].append(environment_base)
        arrays["environment_executed"].append(environment_executed)
        arrays["environment_residual"].append(environment_residual)
        arrays["decoded_base"].append(decoded_base)
        arrays["decoded_executed"].append(decoded_executed)
        arrays["decoded_residual"].append(decoded_residual)
        step_norm = np.linalg.norm(residual, axis=-1)
        mean_norm = np.linalg.norm(mean_action, axis=-1)
        exploration_norm = np.linalg.norm(exploration_action, axis=-1)
        base_norm = np.linalg.norm(base, axis=-1)
        environment_norm = np.linalg.norm(environment_residual, axis=-1)
        for batch_idx in range(residual.shape[0]):
            task_id = int(task_ids[batch_idx])
            trial_id = int(trial_ids[batch_idx])
            item = residual[batch_idx]
            mean_item = mean_action[batch_idx]
            mean_dimension_abs = np.abs(mean_item).mean(axis=0)
            dominant_dimension_idx = int(np.argmax(mean_dimension_abs))
            dominant_dimension = (
                DIMENSION_NAMES[dominant_dimension_idx]
                if dominant_dimension_idx < len(DIMENSION_NAMES)
                else f"dim_{dominant_dimension_idx}"
            )
            rows.append(
                {
                    "worker_pid": worker_pid,
                    "call_idx": call_idx,
                    "batch_idx": batch_idx,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "reset_id": int(reset_ids[batch_idx]),
                    "success": outcomes.get((task_id, trial_id), ""),
                    "action_horizon": int(residual.shape[1]),
                    "sample_l2_mean": float(step_norm[batch_idx].mean()),
                    "sample_l2_max": float(step_norm[batch_idx].max()),
                    "mean_l2_mean": float(mean_norm[batch_idx].mean()),
                    "mean_l2_max": float(mean_norm[batch_idx].max()),
                    "exploration_l2_mean": float(exploration_norm[batch_idx].mean()),
                    "log_std_mean": float(log_std[batch_idx].mean()),
                    "std_mean": float(np.exp(log_std[batch_idx]).mean()),
                    "sample_active_fraction_gt_0.01": float(
                        (step_norm[batch_idx] > 0.01).mean()
                    ),
                    "sample_active_fraction_gt_0.05": float(
                        (step_norm[batch_idx] > 0.05).mean()
                    ),
                    "mean_active_fraction_gt_0.01": float(
                        (mean_norm[batch_idx] > 0.01).mean()
                    ),
                    "mean_active_fraction_gt_0.05": float(
                        (mean_norm[batch_idx] > 0.05).mean()
                    ),
                    "sample_abs_mean": float(np.abs(item).mean()),
                    "mean_abs_mean": float(np.abs(mean_action[batch_idx]).mean()),
                    "exploration_abs_mean": float(
                        np.abs(exploration_action[batch_idx]).mean()
                    ),
                    "sample_saturation_fraction_gt_0.09": float(
                        (np.abs(item) > SATURATION_THRESHOLD).mean()
                    ),
                    "mean_saturation_fraction_gt_0.09": float(
                        (np.abs(mean_item) > SATURATION_THRESHOLD).mean()
                    ),
                    "raw_sample_pressure_fraction_gt_1.472": float(
                        (np.abs(raw_action[batch_idx]) > RAW_PRESSURE_THRESHOLD).mean()
                    ),
                    "raw_mean_pressure_fraction_gt_1.472": float(
                        (np.abs(raw_mean[batch_idx]) > RAW_PRESSURE_THRESHOLD).mean()
                    ),
                    "mean_abs_max": float(np.abs(mean_item).max()),
                    "dominant_dimension": dominant_dimension,
                    "dominant_horizon": int(np.argmax(mean_norm[batch_idx])),
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
                    "mean_residual_to_base_l2_ratio": float(
                        (mean_norm[batch_idx] / (base_norm[batch_idx] + 1e-8)).mean()
                    ),
                    "environment_residual_l2_mean": float(
                        finite_mean(environment_norm[batch_idx])
                    ),
                    "environment_residual_l2_max": float(
                        finite_max(environment_norm[batch_idx])
                    ),
                    "decoded_translation_abs_mean": float(
                        finite_mean(np.abs(decoded_residual[batch_idx, ..., :3]))
                    ),
                    "decoded_rotation_abs_mean": float(
                        finite_mean(np.abs(decoded_residual[batch_idx, ..., 3:6]))
                    )
                    if item.shape[-1] >= 6
                    else float("nan"),
                    "decoded_gripper_abs_mean": float(
                        finite_mean(np.abs(decoded_residual[batch_idx, ..., 6:7]))
                    )
                    if item.shape[-1] >= 7
                    else float("nan"),
                    "environment_translation_abs_mean": float(
                        finite_mean(np.abs(environment_residual[batch_idx, ..., :3]))
                    ),
                    "environment_rotation_abs_mean": float(
                        finite_mean(np.abs(environment_residual[batch_idx, ..., 3:6]))
                    )
                    if item.shape[-1] >= 6
                    else float("nan"),
                    "environment_gripper_abs_mean": float(
                        finite_mean(np.abs(environment_residual[batch_idx, ..., 6:7]))
                    )
                    if item.shape[-1] >= 7
                    else float("nan"),
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
    annotate_episode_time(rows)
    return rows, {
        name: np.concatenate(chunks, axis=0) for name, chunks in arrays.items()
    }


def annotate_episode_time(rows: list[dict[str, object]]) -> None:
    """Add within-trial timing using action-call order as a trajectory proxy.

    LIBERO observations expose stable task/trial/reset identifiers but no
    semantic phase label.  The ordered action calls are therefore retained as
    an exact timeline, while normalized early/middle/late labels are explicitly
    treated as a temporal proxy rather than as inferred task phases.
    """
    grouped: dict[tuple[int, int, int, int, int], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            int(row["worker_pid"]),
            int(row["batch_idx"]),
            int(row["task_id"]),
            int(row["trial_id"]),
            int(row["reset_id"]),
        )
        grouped[key].append(row)

    for values in grouped.values():
        values.sort(key=lambda row: int(row["call_idx"]))
        count = len(values)
        for action_call_index, row in enumerate(values):
            progress = action_call_index / (count - 1) if count > 1 else 0.0
            bin_index = min(int(progress * len(PROGRESS_BIN_NAMES)), 2)
            row["episode_action_call_index"] = action_call_index
            row["episode_action_call_count"] = count
            row["episode_chunk_start_step"] = action_call_index * int(
                row["action_horizon"]
            )
            row["episode_progress_fraction"] = progress
            row["episode_progress_bin"] = PROGRESS_BIN_NAMES[bin_index]


def aggregate_trials(
    rows: list[dict[str, object]],
    pairing: dict[tuple[int, int], str],
) -> list[dict[str, object]]:
    """Aggregate per-call diagnostic records into trial-level evidence."""
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["task_id"]), int(row["trial_id"]))].append(row)

    trial_rows = []
    for (task_id, trial_id), values in sorted(grouped.items()):
        max_pressure_row = max(values, key=lambda row: float(row["mean_abs_max"]))

        def mean(field: str) -> float:
            return float(np.mean([float(row[field]) for row in values]))

        success = next((row["success"] for row in values if row["success"] != ""), "")
        trial_rows.append(
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "reset_id": int(values[0]["reset_id"]),
                "success": success,
                "pairing_transition": pairing.get((task_id, trial_id), ""),
                "num_action_calls": len(values),
                "mean_residual_l2": mean("sample_l2_mean"),
                "max_residual_l2": max(float(row["sample_l2_max"]) for row in values),
                "mean_policy_residual_l2": mean("mean_l2_mean"),
                "max_policy_residual_l2": max(
                    float(row["mean_l2_max"]) for row in values
                ),
                "sample_saturation_fraction": mean(
                    "sample_saturation_fraction_gt_0.09"
                ),
                "mean_saturation_fraction": mean("mean_saturation_fraction_gt_0.09"),
                "raw_sample_pressure_fraction": mean(
                    "raw_sample_pressure_fraction_gt_1.472"
                ),
                "raw_mean_pressure_fraction": mean(
                    "raw_mean_pressure_fraction_gt_1.472"
                ),
                "translation_abs_mean": mean("translation_abs_mean"),
                "rotation_abs_mean": mean("rotation_abs_mean"),
                "gripper_abs_mean": mean("gripper_abs_mean"),
                "residual_to_base_ratio": mean("residual_to_base_l2_ratio"),
                "mean_residual_to_base_ratio": mean("mean_residual_to_base_l2_ratio"),
                "exploration_residual_l2": mean("exploration_l2_mean"),
                "log_std_mean": mean("log_std_mean"),
                "std_mean": mean("std_mean"),
                "sample_active_fraction_gt_0.01": mean(
                    "sample_active_fraction_gt_0.01"
                ),
                "sample_active_fraction_gt_0.05": mean(
                    "sample_active_fraction_gt_0.05"
                ),
                "mean_active_fraction_gt_0.01": mean("mean_active_fraction_gt_0.01"),
                "mean_active_fraction_gt_0.05": mean("mean_active_fraction_gt_0.05"),
                "decoded_translation_abs_mean": mean("decoded_translation_abs_mean"),
                "decoded_rotation_abs_mean": mean("decoded_rotation_abs_mean"),
                "decoded_gripper_abs_mean": mean("decoded_gripper_abs_mean"),
                "environment_translation_abs_mean": mean(
                    "environment_translation_abs_mean"
                ),
                "environment_rotation_abs_mean": mean("environment_rotation_abs_mean"),
                "environment_gripper_abs_mean": mean("environment_gripper_abs_mean"),
                "max_mean_residual_abs": max(
                    float(row["mean_abs_max"]) for row in values
                ),
                "dominant_dimension": max_pressure_row["dominant_dimension"],
                "dominant_horizon": max_pressure_row["dominant_horizon"],
                "peak_episode_action_call_index": max_pressure_row[
                    "episode_action_call_index"
                ],
                "peak_episode_progress_fraction": max_pressure_row[
                    "episode_progress_fraction"
                ],
                "peak_episode_progress_bin": max_pressure_row["episode_progress_bin"],
            }
        )
    return trial_rows


def aggregate_tasks(trial_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize success transitions and residual behavior per LIBERO task."""
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        grouped[int(row["task_id"])].append(row)

    task_rows = []
    for task_id, values in sorted(grouped.items()):
        transitions = [str(row["pairing_transition"]) for row in values]
        paired_transitions = [transition for transition in transitions if transition]
        successes = [float(row["success"]) for row in values if row["success"] != ""]

        def mean(field: str) -> float:
            return float(np.mean([float(row[field]) for row in values]))

        task_rows.append(
            {
                "task_id": task_id,
                "num_trials": len(values),
                "success_rate": float(np.mean(successes)) if successes else "",
                "num_rescue": transitions.count("rescue"),
                "num_harm": transitions.count("harm"),
                "num_preserve": transitions.count("preserve"),
                "num_unresolved": transitions.count("unresolved"),
                "rescue_rate": transitions.count("rescue") / len(paired_transitions)
                if paired_transitions
                else "",
                "harm_rate": transitions.count("harm") / len(paired_transitions)
                if paired_transitions
                else "",
                "mean_policy_residual_l2": mean("mean_policy_residual_l2"),
                "exploration_residual_l2": mean("exploration_residual_l2"),
                "mean_saturation_fraction": mean("mean_saturation_fraction"),
                "raw_mean_pressure_fraction": mean("raw_mean_pressure_fraction"),
                "mean_residual_to_base_ratio": mean("mean_residual_to_base_ratio"),
                "mean_active_fraction_gt_0.01": mean("mean_active_fraction_gt_0.01"),
                "translation_abs_mean": mean("translation_abs_mean"),
                "rotation_abs_mean": mean("rotation_abs_mean"),
                "gripper_abs_mean": mean("gripper_abs_mean"),
                "decoded_translation_abs_mean": mean("decoded_translation_abs_mean"),
                "decoded_rotation_abs_mean": mean("decoded_rotation_abs_mean"),
                "decoded_gripper_abs_mean": mean("decoded_gripper_abs_mean"),
                "environment_translation_abs_mean": mean(
                    "environment_translation_abs_mean"
                ),
                "environment_rotation_abs_mean": mean("environment_rotation_abs_mean"),
                "environment_gripper_abs_mean": mean("environment_gripper_abs_mean"),
            }
        )
    return task_rows


def aggregate_episode_time(
    rows: list[dict[str, object]],
    *,
    per_task: bool = False,
) -> list[dict[str, object]]:
    """Aggregate the exact per-call timeline into temporal proxy bins."""
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            (row["task_id"], row["episode_progress_bin"])
            if per_task
            else (row["episode_progress_bin"],)
        )
        grouped[key].append(row)

    output = []
    order = {name: index for index, name in enumerate(PROGRESS_BIN_NAMES)}
    for key, values in sorted(
        grouped.items(),
        key=lambda item: (
            int(item[0][0]) if per_task else 0,
            order[str(item[0][-1])],
        ),
    ):

        def mean(field: str) -> float:
            return float(np.mean([float(row[field]) for row in values]))

        trial_keys = {
            (row["worker_pid"], row["task_id"], row["trial_id"], row["reset_id"])
            for row in values
        }
        result: dict[str, object] = {}
        if per_task:
            result["task_id"] = int(key[0])
        result.update(
            {
                "episode_progress_bin": key[-1],
                "num_action_calls": len(values),
                "num_trials": len(trial_keys),
                "sample_residual_l2": mean("sample_l2_mean"),
                "mean_policy_residual_l2": mean("mean_l2_mean"),
                "exploration_residual_l2": mean("exploration_l2_mean"),
                "log_std_mean": mean("log_std_mean"),
                "sample_saturation_fraction": mean(
                    "sample_saturation_fraction_gt_0.09"
                ),
                "mean_saturation_fraction": mean("mean_saturation_fraction_gt_0.09"),
                "raw_sample_pressure_fraction": mean(
                    "raw_sample_pressure_fraction_gt_1.472"
                ),
                "raw_mean_pressure_fraction": mean(
                    "raw_mean_pressure_fraction_gt_1.472"
                ),
                "mean_active_fraction_gt_0.01": mean("mean_active_fraction_gt_0.01"),
                "mean_residual_to_base_ratio": mean("mean_residual_to_base_l2_ratio"),
                "translation_abs_mean": mean("translation_abs_mean"),
                "rotation_abs_mean": mean("rotation_abs_mean"),
                "gripper_abs_mean": mean("gripper_abs_mean"),
                "decoded_translation_abs_mean": mean("decoded_translation_abs_mean"),
                "decoded_rotation_abs_mean": mean("decoded_rotation_abs_mean"),
                "decoded_gripper_abs_mean": mean("decoded_gripper_abs_mean"),
                "environment_translation_abs_mean": mean(
                    "environment_translation_abs_mean"
                ),
                "environment_rotation_abs_mean": mean("environment_rotation_abs_mean"),
                "environment_gripper_abs_mean": mean("environment_gripper_abs_mean"),
            }
        )
        output.append(result)
    return output


def high_pressure_trials(
    trial_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Select trials whose deterministic correction pushes against the bound."""
    selected = []
    for row in trial_rows:
        if not (
            float(row["mean_saturation_fraction"]) > 0.10
            or float(row["max_mean_residual_abs"]) > 0.095
        ):
            continue
        selected.append(
            {
                "task_id": row["task_id"],
                "trial_id": row["trial_id"],
                "reset_id": row["reset_id"],
                "success": row["success"],
                "transition": row["pairing_transition"],
                "mean_saturation_fraction": row["mean_saturation_fraction"],
                "max_mean_residual_abs": row["max_mean_residual_abs"],
                "dominant_dimension": row["dominant_dimension"],
                "dominant_horizon": row["dominant_horizon"],
                "peak_episode_action_call_index": row["peak_episode_action_call_index"],
                "peak_episode_progress_fraction": row["peak_episode_progress_fraction"],
                "peak_episode_progress_bin": row["peak_episode_progress_bin"],
            }
        )
    return selected


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
    parser.add_argument("--pairing-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = read_outcomes(args.trials_csv)
    pairing = read_pairing(args.pairing_csv)
    rows, arrays = scalar_rows(args.diagnostic_dir, outcomes)
    write_csv(args.output_dir / "step_records.csv", rows)
    trial_rows = aggregate_trials(rows, pairing)
    write_csv(args.output_dir / "per_trial_residual.csv", trial_rows)
    task_rows = aggregate_tasks(trial_rows)
    write_csv(args.output_dir / "per_task_residual.csv", task_rows)
    episode_time_rows = aggregate_episode_time(rows)
    write_csv(
        args.output_dir / "per_episode_time.csv",
        episode_time_rows,
    )
    write_csv(
        args.output_dir / "per_task_episode_time.csv",
        aggregate_episode_time(rows, per_task=True),
    )
    pressure_rows = high_pressure_trials(trial_rows)
    if pressure_rows:
        write_csv(args.output_dir / "high_pressure_trials.csv", pressure_rows)
    else:
        (args.output_dir / "high_pressure_trials.csv").write_text(
            "task_id,trial_id,reset_id,success,transition,mean_saturation_fraction,"
            "max_mean_residual_abs,dominant_dimension,dominant_horizon,"
            "peak_episode_action_call_index,peak_episode_progress_fraction,"
            "peak_episode_progress_bin\n",
            encoding="utf-8",
        )

    residual = arrays["sample"]
    mean_action = arrays["mean"]
    exploration_action = arrays["exploration"]
    raw_sample = arrays["raw_sample"]
    raw_mean = arrays["raw_mean"]
    log_std = arrays["log_std"]
    base_action = arrays["base"]
    executed_action = arrays["executed"]
    decoded_residual = arrays["decoded_residual"]
    environment_residual = arrays["environment_residual"]
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
                    (np.abs(residual[:, horizon]) > SATURATION_THRESHOLD).mean()
                ),
                "mean_saturation_fraction_gt_0.09": float(
                    (np.abs(mean_action[:, horizon]) > SATURATION_THRESHOLD).mean()
                ),
                "raw_sample_pressure_fraction_gt_1.472": float(
                    (np.abs(raw_sample[:, horizon]) > RAW_PRESSURE_THRESHOLD).mean()
                ),
                "raw_mean_pressure_fraction_gt_1.472": float(
                    (np.abs(raw_mean[:, horizon]) > RAW_PRESSURE_THRESHOLD).mean()
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
                    (np.abs(residual[..., dimension]) > SATURATION_THRESHOLD).mean()
                ),
                "mean_saturation_fraction_gt_0.09": float(
                    (np.abs(mean_action[..., dimension]) > SATURATION_THRESHOLD).mean()
                ),
                "raw_sample_pressure_fraction_gt_1.472": float(
                    (np.abs(raw_sample[..., dimension]) > RAW_PRESSURE_THRESHOLD).mean()
                ),
                "raw_mean_pressure_fraction_gt_1.472": float(
                    (np.abs(raw_mean[..., dimension]) > RAW_PRESSURE_THRESHOLD).mean()
                ),
                "log_std_mean": float(log_std[..., dimension].mean()),
                "std_mean": float(np.exp(log_std[..., dimension]).mean()),
            }
        )
    write_csv(args.output_dir / "per_dimension.csv", dimension_rows)

    if np.isfinite(decoded_residual).any():
        decoded_dimension_rows = []
        for dimension in range(decoded_residual.shape[-1]):
            name = (
                DIMENSION_NAMES[dimension]
                if dimension < len(DIMENSION_NAMES)
                else f"dim_{dimension}"
            )
            values = np.abs(decoded_residual[..., dimension])
            decoded_dimension_rows.append(
                {
                    "dimension": dimension,
                    "name": name,
                    **{
                        f"decoded_abs_{key}": value
                        for key, value in describe(values).items()
                    },
                }
            )
        write_csv(
            args.output_dir / "per_decoded_dimension.csv",
            decoded_dimension_rows,
        )

    if np.isfinite(environment_residual).any():
        environment_dimension_rows = []
        for dimension in range(environment_residual.shape[-1]):
            name = (
                DIMENSION_NAMES[dimension]
                if dimension < len(DIMENSION_NAMES)
                else f"dim_{dimension}"
            )
            values = np.abs(environment_residual[..., dimension])
            environment_dimension_rows.append(
                {
                    "dimension": dimension,
                    "name": name,
                    **{
                        f"environment_abs_{key}": value
                        for key, value in describe(values).items()
                    },
                }
            )
        write_csv(
            args.output_dir / "per_environment_dimension.csv",
            environment_dimension_rows,
        )

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

    transition_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in trial_rows:
        if row["pairing_transition"]:
            transition_groups[str(row["pairing_transition"])].append(row)
    transition_rows = []
    for transition, values in sorted(transition_groups.items()):
        transition_rows.append(
            {
                "transition": transition,
                "num_trials": len(values),
                "mean_saturation_fraction": float(
                    np.mean([row["mean_saturation_fraction"] for row in values])
                ),
                "sample_saturation_fraction": float(
                    np.mean([row["sample_saturation_fraction"] for row in values])
                ),
                "raw_mean_pressure_fraction": float(
                    np.mean([row["raw_mean_pressure_fraction"] for row in values])
                ),
                "raw_sample_pressure_fraction": float(
                    np.mean([row["raw_sample_pressure_fraction"] for row in values])
                ),
                "max_mean_residual_abs": max(
                    float(row["max_mean_residual_abs"]) for row in values
                ),
                "residual_to_base_ratio": float(
                    np.mean([row["residual_to_base_ratio"] for row in values])
                ),
            }
        )
    if transition_rows:
        write_csv(args.output_dir / "transition_conditioned.csv", transition_rows)

    summary = {
        "num_diagnostic_samples": int(residual.shape[0]),
        "action_horizon": int(residual.shape[1]),
        "action_dim": int(residual.shape[2]),
        "sample_residual_l2": describe(horizon_norm),
        "mean_residual_l2": describe(mean_horizon_norm),
        "exploration_residual_l2": describe(exploration_horizon_norm),
        "sample_residual_abs": describe(np.abs(residual)),
        "mean_residual_abs": describe(np.abs(mean_action)),
        "raw_sample_abs": describe(np.abs(raw_sample)),
        "raw_mean_abs": describe(np.abs(raw_mean)),
        "log_std": describe(log_std),
        "std": describe(np.exp(log_std)),
        "sample_active_fraction_gt_0.01": float((horizon_norm > 0.01).mean()),
        "sample_active_fraction_gt_0.05": float((horizon_norm > 0.05).mean()),
        "mean_active_fraction_gt_0.01": float((mean_horizon_norm > 0.01).mean()),
        "mean_active_fraction_gt_0.05": float((mean_horizon_norm > 0.05).mean()),
        "sample_saturation_fraction_gt_0.09": float(
            (np.abs(residual) > SATURATION_THRESHOLD).mean()
        ),
        "mean_saturation_fraction_gt_0.09": float(
            (np.abs(mean_action) > SATURATION_THRESHOLD).mean()
        ),
        "raw_sample_pressure_fraction_gt_1.472": float(
            (np.abs(raw_sample) > RAW_PRESSURE_THRESHOLD).mean()
        ),
        "raw_mean_pressure_fraction_gt_1.472": float(
            (np.abs(raw_mean) > RAW_PRESSURE_THRESHOLD).mean()
        ),
        "num_trials": len(trial_rows),
        "num_high_pressure_trials": len(pressure_rows),
        "episode_time": {
            str(row["episode_progress_bin"]): {
                key: value
                for key, value in row.items()
                if key not in {"episode_progress_bin", "num_action_calls", "num_trials"}
            }
            for row in episode_time_rows
        },
        "base_normalized_ood_fraction": float((np.abs(base_action) > 1.0).mean()),
        "executed_normalized_ood_fraction": float(
            (np.abs(executed_action) > 1.0).mean()
        ),
        "created_normalized_ood_fraction": float(
            ((np.abs(base_action) <= 1.0) & (np.abs(executed_action) > 1.0)).mean()
        ),
    }
    if np.isfinite(environment_residual).any():
        summary["environment_residual_abs"] = describe(np.abs(environment_residual))
        summary["environment_translation_residual_abs"] = describe(
            np.abs(environment_residual[..., :3])
        )
        if environment_residual.shape[-1] >= 6:
            summary["environment_rotation_residual_abs"] = describe(
                np.abs(environment_residual[..., 3:6])
            )
        if environment_residual.shape[-1] >= 7:
            summary["environment_gripper_residual_abs"] = describe(
                np.abs(environment_residual[..., 6:7])
            )
    if np.isfinite(decoded_residual).any():
        summary["decoded_residual_abs"] = describe(np.abs(decoded_residual))
        summary["decoded_translation_residual_abs"] = describe(
            np.abs(decoded_residual[..., :3])
        )
        if decoded_residual.shape[-1] >= 6:
            summary["decoded_rotation_residual_abs"] = describe(
                np.abs(decoded_residual[..., 3:6])
            )
        if decoded_residual.shape[-1] >= 7:
            summary["decoded_gripper_residual_abs"] = describe(
                np.abs(decoded_residual[..., 6:7])
            )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
