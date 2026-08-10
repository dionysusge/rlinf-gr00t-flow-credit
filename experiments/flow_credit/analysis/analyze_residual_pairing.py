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
"""Compare paired fixed-reset evaluations and quantify rescue versus harm."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

KEY = ("task_id", "trial_id")


def mcnemar_exact_pvalue(rescue: int, harm: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired outcomes."""
    discordant = int(rescue) + int(harm)
    if discordant == 0:
        return 1.0
    tail = min(int(rescue), int(harm))
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * probability)


def read_trials(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    """Read one structured fixed-evaluation CSV and enforce unique trial keys."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = {}
    for row in rows:
        key = tuple(int(row[field]) for field in KEY)
        if key in records:
            raise ValueError(f"Duplicate trial {key} in {path}")
        records[key] = row
    if not records:
        raise ValueError(f"No trial records found in {path}")
    return records


def classify(base_success: bool, candidate_success: bool) -> str:
    if base_success and candidate_success:
        return "preserve"
    if not base_success and candidate_success:
        return "rescue"
    if base_success and not candidate_success:
        return "harm"
    return "unresolved"


def analyze(
    base: dict[tuple[int, int], dict[str, str]],
    candidate: dict[tuple[int, int], dict[str, str]],
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Build per-trial, aggregate and per-task paired statistics."""
    if set(base) != set(candidate):
        missing_candidate = sorted(set(base).difference(candidate))
        missing_base = sorted(set(candidate).difference(base))
        raise ValueError(
            "Paired trial sets differ: "
            f"missing_candidate={missing_candidate[:10]}, "
            f"missing_base={missing_base[:10]}"
        )

    pairing_rows = []
    per_task: dict[int, Counter] = defaultdict(Counter)
    total = Counter()
    for task_id, trial_id in sorted(base):
        base_row = base[(task_id, trial_id)]
        candidate_row = candidate[(task_id, trial_id)]
        base_success = bool(int(base_row["success"]))
        candidate_success = bool(int(candidate_row["success"]))
        transition = classify(base_success, candidate_success)
        total[transition] += 1
        per_task[task_id][transition] += 1
        pairing_rows.append(
            {
                "task_id": task_id,
                "trial_id": trial_id,
                "reset_id": int(base_row.get("reset_id", -1)),
                "base_success": int(base_success),
                "candidate_success": int(candidate_success),
                "transition": transition,
                "base_reward": float(base_row.get("reward", "nan")),
                "candidate_reward": float(candidate_row.get("reward", "nan")),
                "base_episode_length": int(base_row.get("episode_length", -1)),
                "candidate_episode_length": int(
                    candidate_row.get("episode_length", -1)
                ),
            }
        )

    trials = len(pairing_rows)
    base_successes = total["preserve"] + total["harm"]
    candidate_successes = total["preserve"] + total["rescue"]
    summary = {
        "num_trials": trials,
        "base_successes": base_successes,
        "candidate_successes": candidate_successes,
        "base_success_rate": base_successes / trials,
        "candidate_success_rate": candidate_successes / trials,
        "success_rate_delta": (candidate_successes - base_successes) / trials,
        "preserve": total["preserve"],
        "rescue": total["rescue"],
        "harm": total["harm"],
        "unresolved": total["unresolved"],
        "net_rescue": total["rescue"] - total["harm"],
        "discordant_pairs": total["rescue"] + total["harm"],
        "mcnemar_exact_pvalue": mcnemar_exact_pvalue(total["rescue"], total["harm"]),
    }

    task_rows = []
    for task_id, counts in sorted(per_task.items()):
        task_trials = sum(counts.values())
        task_base = counts["preserve"] + counts["harm"]
        task_candidate = counts["preserve"] + counts["rescue"]
        task_rows.append(
            {
                "task_id": task_id,
                "num_trials": task_trials,
                "base_success_rate": task_base / task_trials,
                "candidate_success_rate": task_candidate / task_trials,
                "success_rate_delta": (task_candidate - task_base) / task_trials,
                "preserve": counts["preserve"],
                "rescue": counts["rescue"],
                "harm": counts["harm"],
                "unresolved": counts["unresolved"],
                "net_rescue": counts["rescue"] - counts["harm"],
            }
        )
    return pairing_rows, summary, task_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairing_rows, summary, task_rows = analyze(
        read_trials(args.base),
        read_trials(args.candidate),
    )
    write_csv(args.output_dir / "pairing.csv", pairing_rows)
    write_csv(args.output_dir / "per_task.csv", task_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
