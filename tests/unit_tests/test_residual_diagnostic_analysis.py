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
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "flow_credit"
    / "analysis"
    / "analyze_residual_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("residual_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_diagnostic_analysis_separates_mean_exploration_and_ood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic_dir = tmp_path / "diagnostics"
    diagnostic_dir.mkdir()
    residual = np.full((1, 2, 3), 0.095, dtype=np.float32)
    mean = np.full((1, 2, 3), 0.05, dtype=np.float32)
    raw_sample = np.full((1, 2, 3), 1.6, dtype=np.float32)
    raw_mean = np.full((1, 2, 3), 1.0, dtype=np.float32)
    base = np.full((1, 2, 3), 0.98, dtype=np.float32)
    executed = base + residual
    environment_base = np.full((1, 2, 3), 0.2, dtype=np.float32)
    environment_residual = np.full((1, 2, 3), 0.01, dtype=np.float32)
    np.savez_compressed(
        diagnostic_dir / "residual_000000_pid123.npz",
        residual_action=residual,
        residual_raw_action=raw_sample,
        residual_mean=raw_mean,
        residual_mean_action=mean,
        residual_exploration_action=residual - mean,
        residual_log_std=np.full((1, 2, 3), -2.5, dtype=np.float32),
        base_action=base,
        executed_normalized_action=executed,
        environment_base_action=environment_base,
        environment_executed_action=environment_base + environment_residual,
        environment_residual_action=environment_residual,
        task_ids=np.array([0]),
        trial_ids=np.array([1]),
        reset_ids=np.array([42]),
        call_idx=np.array(0),
    )

    rows, arrays = MODULE.scalar_rows(diagnostic_dir, {(0, 1): 1})

    assert rows[0]["success"] == 1
    assert rows[0]["sample_saturation_fraction_gt_0.09"] == 1.0
    assert rows[0]["mean_saturation_fraction_gt_0.09"] == 0.0
    assert rows[0]["raw_sample_pressure_fraction_gt_1.472"] == 1.0
    assert rows[0]["raw_mean_pressure_fraction_gt_1.472"] == 0.0
    assert rows[0]["created_normalized_ood_fraction"] == 1.0
    assert rows[0]["mean_l2_mean"] == pytest.approx(np.sqrt(3) * 0.05)
    assert rows[0]["environment_residual_l2_mean"] == pytest.approx(np.sqrt(3) * 0.01)
    assert rows[0]["episode_action_call_index"] == 0
    assert rows[0]["episode_progress_bin"] == "early"
    assert np.allclose(arrays["exploration"], residual - mean)
    assert np.allclose(arrays["raw_sample"], raw_sample)
    assert np.allclose(arrays["raw_mean"], raw_mean)

    trials = MODULE.aggregate_trials(rows, {(0, 1): "rescue"})
    assert trials[0]["pairing_transition"] == "rescue"
    assert trials[0]["sample_saturation_fraction"] == 1.0
    assert trials[0]["raw_sample_pressure_fraction"] == 1.0

    trials_csv = tmp_path / "trials.csv"
    trials_csv.write_text(
        "task_id,trial_id,reset_id,success\n0,1,42,1\n",
        encoding="utf-8",
    )
    pairing_csv = tmp_path / "pairing.csv"
    pairing_csv.write_text(
        "task_id,trial_id,transition\n0,1,rescue\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--diagnostic-dir",
            str(diagnostic_dir),
            "--trials-csv",
            str(trials_csv),
            "--pairing-csv",
            str(pairing_csv),
            "--output-dir",
            str(output_dir),
        ],
    )

    MODULE.main()

    assert (output_dir / "per_trial_residual.csv").is_file()
    assert (output_dir / "per_task_residual.csv").is_file()
    assert (output_dir / "per_episode_time.csv").is_file()
    assert (output_dir / "per_task_episode_time.csv").is_file()
    assert (output_dir / "per_decoded_dimension.csv").is_file()
    assert (output_dir / "per_environment_dimension.csv").is_file()
    assert (output_dir / "high_pressure_trials.csv").is_file()
    assert (output_dir / "transition_conditioned.csv").is_file()
    assert "rescue" in (output_dir / "per_trial_residual.csv").read_text(
        encoding="utf-8"
    )


def test_high_pressure_trial_selection() -> None:
    trial = {
        "task_id": 3,
        "trial_id": 27,
        "reset_id": 42,
        "success": 0,
        "pairing_transition": "unresolved",
        "mean_saturation_fraction": 0.11,
        "max_mean_residual_abs": 0.096,
        "dominant_dimension": "drx",
        "dominant_horizon": 0,
        "peak_episode_action_call_index": 2,
        "peak_episode_progress_fraction": 0.5,
        "peak_episode_progress_bin": "middle",
    }

    selected = MODULE.high_pressure_trials([trial])

    assert selected == [
        {
            "task_id": 3,
            "trial_id": 27,
            "reset_id": 42,
            "success": 0,
            "transition": "unresolved",
            "mean_saturation_fraction": 0.11,
            "max_mean_residual_abs": 0.096,
            "dominant_dimension": "drx",
            "dominant_horizon": 0,
            "peak_episode_action_call_index": 2,
            "peak_episode_progress_fraction": 0.5,
            "peak_episode_progress_bin": "middle",
        }
    ]


def test_episode_timeline_uses_order_within_each_trial() -> None:
    rows = [
        {
            "worker_pid": 10,
            "batch_idx": 0,
            "task_id": 1,
            "trial_id": 2,
            "reset_id": 3,
            "call_idx": call_idx,
            "action_horizon": 16,
        }
        for call_idx in (9, 3, 6)
    ]

    MODULE.annotate_episode_time(rows)

    ordered = sorted(rows, key=lambda row: row["call_idx"])
    assert [row["episode_action_call_index"] for row in ordered] == [0, 1, 2]
    assert [row["episode_progress_bin"] for row in ordered] == [
        "early",
        "middle",
        "late",
    ]
    assert [row["episode_chunk_start_step"] for row in ordered] == [0, 16, 32]
