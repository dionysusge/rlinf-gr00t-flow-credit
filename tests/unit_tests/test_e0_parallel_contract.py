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
SCRIPT = ROOT / "experiments/flow_credit/scripts/run_n17_residual_e0_fixed500.sh"
CONFIG = (
    ROOT
    / "examples/embodiment/config/libero_spatial_n17_residual_fixed_eval_gpu45.yaml"
)
EVALUATOR = ROOT / "examples/embodiment/eval_embodied_agent_fixed.py"


def test_e0_keeps_two_gpu_ray_parallelism() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")

    assert "CONFIG_NAME=libero_spatial_n17_residual_fixed_eval_gpu45" in script
    assert "eval_embodied_agent_fixed.py" in script
    assert "RLINF_FORCE_LOCAL_RAY=1" in script
    assert "actor: 4-5" in config
    assert "rollout: 4-5" in config
    assert "env: 4-5" in config


def test_e0_uses_non_overlapping_slices_and_bounded_retries() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "declare -a OFFSETS=(0 100 200 300 400)" in script
    assert 'RLINF_TIMEOUT="${E0_COLLECTIVE_TIMEOUT_MINUTES:-15}"' in script
    assert 'E0_MAX_ATTEMPTS="${E0_MAX_ATTEMPTS:-2}"' in script
    assert "timeout --signal=TERM --kill-after=120s" in script
    assert "n17_residual_e0.lock" in script


def test_e0_fixed100_cycles_twenty_envs_over_five_epochs() -> None:
    config = CONFIG.read_text(encoding="utf-8")

    eval_config = config.split("  eval:\n", maxsplit=1)[1].split(
        "\nrollout:", maxsplit=1
    )[0]
    assert "total_num_envs: 20" in eval_config
    assert "auto_reset: true" in eval_config
    assert "rollout_epoch: 5" in eval_config


def test_fixed_evaluator_validates_unique_reset_slice_records() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    assert 'cfg.env.eval.get(\n                "eval_reset_limit"' in source
    assert "validate_fixed_trial_records(trial_records, expected=expected)" in source
    assert 'unique_trials = {(record["task_id"], record["trial_id"])' in source
    assert 'actual = int(metrics.get("num_trajectories", -1))' not in source


def test_fixed_evaluator_orders_worker_cleanup_before_ray_shutdown() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")

    env_position = source.index('("env", env_group)')
    rollout_position = source.index('("rollout", rollout_group)')
    actor_position = source.index('("actor", actor_group)')
    shutdown_position = source.index("ray.shutdown()")
    assert env_position < rollout_position < actor_position < shutdown_position
    assert "signal.SIGTERM" in source
