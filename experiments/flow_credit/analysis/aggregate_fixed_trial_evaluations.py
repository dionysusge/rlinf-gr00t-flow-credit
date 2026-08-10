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
"""Aggregate structured fixed-reset evaluation directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_input(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Inputs must use SET_NAME=PATH") from error
    return label, Path(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    set_rows = []
    for set_name, eval_dir in args.input:
        with (eval_dir / "trials.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"No trials in {eval_dir / 'trials.csv'}")
        for row in rows:
            trials.append({"set": set_name, **row})
        successes = sum(int(row["success"]) for row in rows)
        set_rows.append(
            {
                "set": set_name,
                "successes": successes,
                "num_trials": len(rows),
                "success_rate": successes / len(rows),
                "reward": sum(float(row["reward"]) for row in rows) / len(rows),
                "episode_length": sum(int(row["episode_length"]) for row in rows)
                / len(rows),
            }
        )

    keys = [(row["task_id"], row["trial_id"]) for row in trials]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (task_id, trial_id) across evaluation sets")
    write_csv(args.output_dir / "trials.csv", trials)
    write_csv(args.output_dir / "per_set_summary.csv", set_rows)
    successes = sum(int(row["success"]) for row in trials)
    summary = {
        "successes": successes,
        "num_trials": len(trials),
        "success_rate": successes / len(trials),
        "reward": sum(float(row["reward"]) for row in trials) / len(trials),
        "episode_length": sum(int(row["episode_length"]) for row in trials)
        / len(trials),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
