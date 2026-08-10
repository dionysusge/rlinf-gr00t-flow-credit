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


def test_diagnostic_analysis_separates_mean_exploration_and_ood(tmp_path: Path) -> None:
    diagnostic_dir = tmp_path / "diagnostics"
    diagnostic_dir.mkdir()
    residual = np.full((1, 2, 3), 0.095, dtype=np.float32)
    mean = np.full((1, 2, 3), 0.05, dtype=np.float32)
    base = np.full((1, 2, 3), 0.98, dtype=np.float32)
    executed = base + residual
    np.savez_compressed(
        diagnostic_dir / "residual_000000_pid123.npz",
        residual_action=residual,
        residual_mean_action=mean,
        residual_exploration_action=residual - mean,
        residual_log_std=np.full((1, 2, 3), -2.5, dtype=np.float32),
        base_action=base,
        executed_normalized_action=executed,
        task_ids=np.array([0]),
        trial_ids=np.array([1]),
        reset_ids=np.array([42]),
        call_idx=np.array(0),
    )

    rows, arrays = MODULE.scalar_rows(diagnostic_dir, {(0, 1): 1})

    assert rows[0]["success"] == 1
    assert rows[0]["sample_saturation_fraction_gt_0.09"] == 1.0
    assert rows[0]["created_normalized_ood_fraction"] == 1.0
    assert rows[0]["mean_l2_mean"] == pytest.approx(np.sqrt(3) * 0.05)
    assert np.allclose(arrays["exploration"], residual - mean)
