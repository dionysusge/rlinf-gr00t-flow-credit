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

"""Structured trial artifacts shared by direct fixed evaluators."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def to_finite_tensor(value: Any) -> torch.Tensor:
    """Copy a value to CPU and reject non-finite evidence."""
    tensor = torch.as_tensor(value).detach().cpu()
    if not torch.isfinite(tensor.float()).all():
        raise FloatingPointError("Non-finite fixed-evaluation value")
    return tensor


def episode_records(
    infos_list: list[dict[str, Any]],
    *,
    label: str,
    checkpoint: str,
) -> list[dict[str, Any]]:
    """Extract only unique completed trials selected by LiberoEnv's count mask."""
    if not infos_list:
        return []
    final_wrapper = infos_list[-1]
    count_mask = final_wrapper.get("_final_info")
    final_info = final_wrapper.get("final_info")
    if count_mask is None or not isinstance(final_info, dict):
        return []
    episode = final_info.get("episode")
    if not isinstance(episode, dict):
        return []

    required = {
        "task_id",
        "trial_id",
        "reset_id",
        "success_once",
        "return",
        "reward",
        "episode_len",
    }
    missing = required.difference(episode)
    if missing:
        raise KeyError(f"Final LIBERO episode metrics are missing {sorted(missing)}")

    mask = to_finite_tensor(count_mask).bool().reshape(-1)
    fields = {name: to_finite_tensor(episode[name]).reshape(-1) for name in required}
    lengths = {name: int(value.numel()) for name, value in fields.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != mask.numel():
        raise ValueError(
            f"Misaligned fixed-evaluation episode fields: mask={mask.numel()}, "
            f"fields={lengths}"
        )

    records = []
    for index in mask.nonzero(as_tuple=False).flatten().tolist():
        records.append(
            {
                "task_id": int(fields["task_id"][index]),
                "trial_id": int(fields["trial_id"][index]),
                "reset_id": int(fields["reset_id"][index]),
                "success": int(bool(fields["success_once"][index])),
                "return": float(fields["return"][index]),
                "reward": float(fields["reward"][index]),
                "episode_length": int(fields["episode_len"][index]),
                "model": label,
                "checkpoint": checkpoint,
            }
        )
    return records


def write_trial_records(output_dir: Path, records: list[dict[str, Any]]) -> None:
    """Write artifacts compatible with the existing E0 aggregation tools."""
    if not records:
        raise ValueError("Fixed evaluation produced no trial records")
    records.sort(key=lambda row: (row["task_id"], row["trial_id"]))
    keys = [(row["task_id"], row["trial_id"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Fixed evaluation produced duplicate task/trial pairs")
    with (output_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
