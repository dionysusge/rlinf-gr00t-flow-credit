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
"""Upload compact flow-credit metrics, tables and evidence files to W&B."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

STRUCTURED_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".yaml", ".yml"}


def coerce_scalar(value: str) -> object:
    """Convert a CSV string to a stable numeric scalar when possible."""
    stripped = value.strip()
    if stripped == "":
        return ""
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        try:
            return float(stripped)
        except ValueError:
            return value


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: coerce_scalar(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_json(
    value: Any,
    *,
    prefix: str = "",
) -> dict[str, int | float | str | bool]:
    """Flatten scalar JSON leaves into slash-delimited W&B metric names."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_json(child, prefix=child_prefix))
    elif isinstance(value, (str, bool, int, float)) or value is None:
        flattened[prefix] = "" if value is None else value
    return flattened


def prefixed_row(prefix: str, row: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}/{key}": value for key, value in row.items()}


def collect_events(kind: str, root: Path) -> tuple[list[dict[str, object]], str | None]:
    """Return ordered metric events and the preferred x-axis metric."""
    if kind == "training":
        return [], None

    if kind == "e0":
        event = {}
        for prefix, relative in (
            ("e0/all500", "aggregate/summary.json"),
            ("e0/heldout_BtoE", "aggregate_heldout_BtoE/summary.json"),
            ("e0/repeatability", "E0_REPEATABILITY.json"),
        ):
            path = root / relative
            if path.is_file():
                event.update(flatten_json(read_json(path), prefix=prefix))
        return [event], None

    if kind == "checkpoint_sweep":
        rows = read_csv(root / "checkpoint_summary.csv")
        events = []
        for row in rows:
            event = prefixed_row("checkpoint", row)
            diagnostic_summary = (
                root / f"step{int(row['step'])}" / "residual_analysis" / "summary.json"
            )
            if diagnostic_summary.is_file():
                event.update(
                    flatten_json(
                        read_json(diagnostic_summary),
                        prefix="checkpoint/residual",
                    )
                )
            events.append(event)
        return events, "checkpoint/environment_transitions"

    if kind == "strength_curve":
        rows = read_csv(root / "strength_curve.csv")
        events = []
        for row in rows:
            event = prefixed_row("strength", row)
            lambda_name = str(row["lambda"]).replace(".", "p")
            lambda_root = root / f"lambda_{lambda_name}"
            for set_name in ("setA", "setB", "setC", "setD", "setE"):
                diagnostic_summary = (
                    lambda_root / set_name / "residual_analysis" / "summary.json"
                )
                if diagnostic_summary.is_file():
                    event.update(
                        flatten_json(
                            read_json(diagnostic_summary),
                            prefix=f"strength/{set_name}/residual",
                        )
                    )
            events.append(event)
        return events, "strength/lambda"

    if kind == "fullppo":
        event = {}
        for prefix, relative in (
            ("fullppo/all500", "aggregate/summary.json"),
            ("fullppo/pairing", "pairing/summary.json"),
            ("fullppo/heldout_BtoE", "aggregate_heldout_BtoE/summary.json"),
            ("fullppo/heldout_pairing", "heldout_BtoE_pairing/summary.json"),
        ):
            path = root / relative
            if path.is_file():
                event.update(flatten_json(read_json(path), prefix=prefix))
        return [event], None

    raise ValueError(f"Unsupported evidence kind: {kind}")


def table_paths(kind: str, root: Path) -> list[tuple[str, Path]]:
    """Select compact research tables that should be browsable in W&B."""
    relative_paths: dict[str, tuple[str, ...]] = {
        "training": (),
        "e0": (
            "aggregate/trials.csv",
            "setA_repeatability/pairing.csv",
        ),
        "checkpoint_sweep": (
            "checkpoint_summary.csv",
            "checkpoint_selection_ranking.csv",
        ),
        "strength_curve": ("strength_curve.csv",),
        "fullppo": (
            "pairing/pairing.csv",
            "heldout_BtoE_pairing/pairing.csv",
        ),
    }
    selected = []
    for relative in relative_paths[kind]:
        path = root / relative
        if path.is_file():
            selected.append((relative.removesuffix(".csv").replace("/", "_"), path))

    diagnostic_tables = (
        "per_trial_residual.csv",
        "per_task_residual.csv",
        "per_episode_time.csv",
        "per_task_episode_time.csv",
        "per_horizon.csv",
        "per_dimension.csv",
        "per_decoded_dimension.csv",
        "per_environment_dimension.csv",
        "high_pressure_trials.csv",
        "transition_conditioned.csv",
        "per_task.csv",
    )
    if kind in {"checkpoint_sweep", "strength_curve", "fullppo"}:
        for filename in diagnostic_tables:
            for path in sorted(root.rglob(filename)):
                relative = path.relative_to(root).as_posix()
                selected.append((relative.removesuffix(".csv").replace("/", "_"), path))
    return selected


def structured_files(root: Path) -> list[Path]:
    """List compact evidence files while excluding raw diagnostics and logs."""
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in STRUCTURED_SUFFIXES:
            continue
        relative_parts = path.relative_to(root).parts
        if "residual_diagnostics" in relative_parts or "wandb" in relative_parts:
            continue
        if path.name.endswith(".log"):
            continue
        files.append(path)
    return sorted(files)


def artifact_files(kind: str, root: Path) -> list[Path]:
    """Select artifact files, keeping training checkpoints out of W&B."""
    if kind != "training":
        return structured_files(root)
    selected = []
    for filename in (
        "run_manifest.json",
        "resolved_config.yaml",
        "launch_config.yaml",
        "git_commit.txt",
        "git_status.txt",
        "e0_evaluation_root.txt",
        "e0_repeatability.json",
    ):
        path = root / filename
        if path.is_file():
            selected.append(path)
    return selected


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def provenance_config(kind: str, root: Path) -> dict[str, object]:
    """Collect compact run provenance written by the experiment scripts."""
    config: dict[str, object] = {
        "evidence_kind": kind,
        "evidence_root": str(root),
    }
    for key, filename in (
        ("git_commit", "git_commit.txt"),
        ("source_checkpoint", "checkpoint.txt"),
        ("checkpoint_root", "checkpoint_root.txt"),
        ("training_run_dir", "training_run_dir.txt"),
        ("e0_evaluation_root", "e0_evaluation_root.txt"),
    ):
        path = root / filename
        if path.is_file():
            config[key] = path.read_text(encoding="utf-8").strip()
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=("training", "e0", "checkpoint_sweep", "strength_curve", "fullppo"),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--group", default=os.environ.get("WANDB_RUN_GROUP"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    if not args.project:
        raise ValueError("W&B project is required via --project or WANDB_PROJECT")

    import wandb

    events, step_metric = collect_events(args.kind, args.root)
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        job_type=args.kind,
        name=args.name,
        id=args.run_id,
        resume="allow",
        config=provenance_config(args.kind, args.root),
        dir=str(args.root),
    )
    try:
        if step_metric is not None:
            namespace = step_metric.split("/", 1)[0]
            wandb.define_metric(step_metric)
            wandb.define_metric(f"{namespace}/*", step_metric=step_metric)
        for event in events:
            if event:
                wandb.log(event)

        for table_name, path in table_paths(args.kind, args.root):
            rows = read_csv(path)
            if not rows:
                continue
            columns = list(rows[0])
            data = [[row[column] for column in columns] for row in rows]
            wandb.log({f"tables/{table_name}": wandb.Table(columns=columns, data=data)})

        artifact = wandb.Artifact(
            name=safe_name(f"{args.kind}-{args.run_id}-evidence"),
            type="flow-credit-evidence",
            metadata={"kind": args.kind, "root": str(args.root)},
        )
        files = artifact_files(args.kind, args.root)
        for path in files:
            artifact.add_file(
                str(path),
                name=path.relative_to(args.root).as_posix(),
            )
        if files:
            run.log_artifact(artifact)
        run.summary["evidence/structured_file_count"] = len(files)
        run.summary["evidence/root"] = str(args.root)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
