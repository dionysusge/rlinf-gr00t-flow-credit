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

import pytest
import torch

MODULE_PATH = Path(__file__).parents[2] / "rlinf" / "utils" / "metric_utils.py"
SPEC = importlib.util.spec_from_file_location("metric_utils", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
compute_evaluate_metrics = MODULE.compute_evaluate_metrics


def test_evaluation_metadata_is_not_averaged_into_scalar_metrics() -> None:
    metrics = compute_evaluate_metrics(
        [
            {
                "task_id": torch.tensor([0, 1]),
                "trial_id": torch.tensor([2, 3]),
                "reset_id": torch.tensor([2, 53]),
                "success_once": torch.tensor([True, False]),
                "reward": torch.tensor([0.2, 0.4]),
            }
        ]
    )

    assert "task_id" not in metrics
    assert "trial_id" not in metrics
    assert "reset_id" not in metrics
    assert metrics["success_once"] == pytest.approx(0.5)
    assert metrics["reward"] == pytest.approx(0.3)
    assert metrics["num_trajectories"] == 2
