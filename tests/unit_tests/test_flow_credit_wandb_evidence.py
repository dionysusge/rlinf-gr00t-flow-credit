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

import importlib.util
import json
import types
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "flow_credit"
    / "analysis"
    / "log_evidence_to_wandb.py"
)
SPEC = importlib.util.spec_from_file_location("flow_credit_wandb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_checkpoint_events_include_residual_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "checkpoint_summary.csv").write_text(
        "step,environment_transitions,success_rate,rescue,harm\n30,122880,0.9,3,1\n",
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "step30" / "residual_analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "summary.json").write_text(
        json.dumps(
            {
                "mean_saturation_fraction_gt_0.09": 0.2,
                "raw_mean_pressure_fraction_gt_1.472": 0.1,
                "num_high_pressure_trials": 4,
            }
        ),
        encoding="utf-8",
    )

    events, step_metric = MODULE.collect_events("checkpoint_sweep", tmp_path)

    assert step_metric == "checkpoint/environment_transitions"
    assert events == [
        {
            "checkpoint/step": 30,
            "checkpoint/environment_transitions": 122880,
            "checkpoint/success_rate": 0.9,
            "checkpoint/rescue": 3,
            "checkpoint/harm": 1,
            "checkpoint/residual/mean_saturation_fraction_gt_0.09": 0.2,
            "checkpoint/residual/raw_mean_pressure_fraction_gt_1.472": 0.1,
            "checkpoint/residual/num_high_pressure_trials": 4,
        }
    ]


def test_structured_artifact_excludes_raw_diagnostics(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("{}\n", encoding="utf-8")
    diagnostic_dir = tmp_path / "residual_diagnostics"
    diagnostic_dir.mkdir()
    (diagnostic_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    (diagnostic_dir / "residual_000.npz").write_bytes(b"raw")
    (tmp_path / "evaluation.log").write_text("large log\n", encoding="utf-8")

    files = MODULE.structured_files(tmp_path)

    assert files == [tmp_path / "summary.json"]


def test_checkpoint_tables_include_temporal_task_and_decoded_action_views(
    tmp_path: Path,
) -> None:
    analysis_dir = tmp_path / "step30" / "residual_analysis"
    analysis_dir.mkdir(parents=True)
    for filename in (
        "per_task_residual.csv",
        "per_episode_time.csv",
        "per_task_episode_time.csv",
        "per_decoded_dimension.csv",
        "per_environment_dimension.csv",
    ):
        (analysis_dir / filename).write_text("value\n1\n", encoding="utf-8")
    pairing_dir = tmp_path / "step30" / "pairing"
    pairing_dir.mkdir()
    (pairing_dir / "per_task.csv").write_text("task_id\n0\n", encoding="utf-8")

    tables = dict(MODULE.table_paths("checkpoint_sweep", tmp_path))

    assert "step30_residual_analysis_per_task_residual" in tables
    assert "step30_residual_analysis_per_episode_time" in tables
    assert "step30_residual_analysis_per_task_episode_time" in tables
    assert "step30_residual_analysis_per_decoded_dimension" in tables
    assert "step30_residual_analysis_per_environment_dimension" in tables
    assert "step30_pairing_per_task" in tables


def test_training_artifact_excludes_checkpoint_metadata(tmp_path: Path) -> None:
    (tmp_path / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_30"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}\n", encoding="utf-8")

    files = MODULE.artifact_files("training", tmp_path)

    assert files == [tmp_path / "run_manifest.json"]


def test_e0_event_flattens_all_evidence_groups(tmp_path: Path) -> None:
    for relative, payload in (
        ("aggregate/summary.json", {"success_rate": 0.888}),
        ("aggregate_heldout_BtoE/summary.json", {"success_rate": 0.89}),
        ("base/aggregate/summary.json", {"success_rate": 0.9}),
        ("base_zero_pairing/summary.json", {"rescue": 0, "harm": 0}),
        ("E0_REPEATABILITY.json", {"repeatable": True, "harm": 0}),
        (
            "historical_reference.json",
            {"historical_unseeded_successes": 444},
        ),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    events, step_metric = MODULE.collect_events("e0", tmp_path)

    assert step_metric is None
    assert events[0]["e0/all500/success_rate"] == 0.888
    assert events[0]["e0/heldout_BtoE/success_rate"] == 0.89
    assert events[0]["e0/base_all500/success_rate"] == 0.9
    assert events[0]["e0/base_zero_pairing/rescue"] == 0
    assert events[0]["e0/repeatability/repeatable"] is True
    assert events[0]["e0/historical_reference/historical_unseeded_successes"] == 444


def test_main_logs_metrics_tables_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    aggregate = tmp_path / "aggregate"
    aggregate.mkdir()
    (aggregate / "summary.json").write_text(
        json.dumps({"success_rate": 0.888}), encoding="utf-8"
    )
    (aggregate / "trials.csv").write_text(
        "task_id,trial_id,success\n0,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "git_commit.txt").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "e0_manifest.json").write_text(
        json.dumps({"rollout_seed": 1234, "env_seed": 0}), encoding="utf-8"
    )

    class FakeArtifact:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.files = []

        def add_file(self, path, name):
            self.files.append((path, name))

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.artifacts = []
            self.finished = False

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self):
            self.finished = True

    fake_run = FakeRun()
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.logged = []
    fake_wandb.defined = []
    fake_wandb.init_kwargs = None

    def init(**kwargs):
        fake_wandb.init_kwargs = kwargs
        return fake_run

    fake_wandb.init = init
    fake_wandb.log = lambda data: fake_wandb.logged.append(data)
    fake_wandb.define_metric = lambda *args, **kwargs: fake_wandb.defined.append(
        (args, kwargs)
    )
    fake_wandb.Table = lambda **kwargs: ("table", kwargs)
    fake_wandb.Artifact = lambda **kwargs: FakeArtifact(**kwargs)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--kind",
            "e0",
            "--root",
            str(tmp_path),
            "--name",
            "e0-test",
            "--run-id",
            "e0-test-id",
            "--project",
            "test-project",
        ],
    )

    MODULE.main()

    assert fake_wandb.init_kwargs["config"]["git_commit"] == "abc123"
    assert fake_wandb.init_kwargs["config"]["e0_experiment"]["rollout_seed"] == 1234
    assert any("e0/all500/success_rate" in event for event in fake_wandb.logged)
    assert any("tables/aggregate_trials" in event for event in fake_wandb.logged)
    assert len(fake_run.artifacts) == 1
    assert fake_run.finished is True
