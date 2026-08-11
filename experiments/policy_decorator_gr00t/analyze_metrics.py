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

"""Summarize Policy Decorator JSONL evidence without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def _numeric_series(records: list[dict]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values[key].append(float(value))
    return values


def analyze(root: Path) -> tuple[Path, Path]:
    """Write a CSV metric table and a compact JSON run summary."""
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "metric_summary.csv"
    json_path = analysis_dir / "run_summary.json"
    rows = []
    run_summary = {}
    for metrics_path in sorted(root.glob("metrics_*.jsonl")):
        mode = metrics_path.stem.removeprefix("metrics_")
        records = _load_records(metrics_path)
        series = _numeric_series(records)
        for metric, values in sorted(series.items()):
            rows.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "count": len(values),
                    "first": values[0],
                    "last": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
        summary_path = root / f"summary_{mode}.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                run_summary[mode] = json.load(handle)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode",
                "metric",
                "count",
                "first",
                "last",
                "mean",
                "min",
                "max",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    csv_path, json_path = analyze(args.result_root.expanduser().resolve())
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
