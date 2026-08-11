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

from pathlib import Path

ROOT = Path(__file__).parents[2]
PPO_CONFIG = (
    ROOT / "examples/embodiment/config/"
    "libero_spatial_residual_ppo_gr00t_n1d7_h200_gpu01.yaml"
)
PPO_SCRIPT = (
    ROOT / "experiments/flow_credit/scripts/run_n17_residual_ppo_train_gpu01.sh"
)
PD_SCRIPT = ROOT / "experiments/policy_decorator_gr00t/run.sh"


def test_residual_ppo_uses_physical_gpu_zero_and_one() -> None:
    config = PPO_CONFIG.read_text(encoding="utf-8")
    script = PPO_SCRIPT.read_text(encoding="utf-8")

    assert "actor: 0-1" in config
    assert "rollout: 0-1" in config
    assert "env: 0-1" in config
    assert 'RUN_ID="n17_residual_ppo_a01_gpu01_${STAMP}"' in script
    assert 'echo "GPUs:                0,1"' in script
    assert '"selected_gpu_indices": [0, 1]' in script


def test_policy_decorator_defaults_to_physical_gpu_zero_and_one() -> None:
    script = PD_SCRIPT.read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="${PD_GPUS:-0,1}"' in script
